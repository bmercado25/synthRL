"""Step 2b training: cross-pitch reconstruction on the NSynth train split.

    .venv/bin/python -m neural.train --steps 40000

Each step samples (source, target) notes from one training instrument and
trains the model to rebuild the target from the source, at the target's pitch
and velocity. Validation uses fixed pairs from the NSynth valid split.

Two checkpoints are written:
  <out>            the best validation loss so far (what neural.evaluate loads)
  <out>_last.pt    the latest state, with optimizer and scheduler, for --resume

The per-step pair sampling and noise seed depend only on the step number, so a
resumed run draws the same batches it would have drawn without the interruption.
"""

import argparse
import time
from pathlib import Path

import torch

from . import data
from .model import NeuralSource, mr_stft_loss

DEFAULT_OUT = data.ROOT.parent / "checkpoints" / "nsynth_2b_full.pt"


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def to_tensors(device, src, tgt, midi, vel):
    return (torch.as_tensor(src, device=device), torch.as_tensor(tgt, device=device),
            torch.as_tensor(midi, device=device), torch.as_tensor(vel, device=device))


@torch.no_grad()
def validate(model, device, val_pairs, batch=8):
    model.eval()
    total, count = 0.0, 0
    for i in range(0, len(val_pairs), batch):
        chunk = val_pairs[i:i + batch]
        src, tgt, midi, vel = to_tensors(device, *data.batch_from_pairs(chunk))
        z = model.encode(src)
        pred = model.render(z, midi, vel, noise_seed=0)
        total += mr_stft_loss(pred, tgt).item() * len(chunk)
        count += len(chunk)
    model.train()
    return total / count


def last_path(out):
    return out.with_name(out.stem + "_last.pt")


def train(steps, batch, lr, out, seed, log_every, eval_every, resume):
    torch.manual_seed(seed)
    device = pick_device()
    sampler = data.PairSampler(data.load_split("train"))
    val_pairs = data.sample_pairs(data.load_split("valid"), count=128, seed=1234)
    print(f"device {device}: {len(sampler.groups)} training instruments, "
          f"{len(val_pairs)} validation pairs", flush=True)

    model = NeuralSource().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    start_step, best = 0, float("inf")
    out.parent.mkdir(parents=True, exist_ok=True)

    if resume and last_path(out).exists():
        ck = torch.load(last_path(out), map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start_step, best = ck["step"], ck["best"]
        print(f"resumed from step {start_step} (best validation {best:.4f})", flush=True)

    def save_last(step):
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "step": step, "best": best,
                    "seed": seed}, last_path(out))

    start = time.time()
    for step in range(start_step + 1, steps + 1):
        pairs = sampler.sample(count=batch, seed=seed * 1_000_003 + step)
        src, tgt, midi, vel = to_tensors(device, *data.batch_from_pairs(pairs))
        z = model.encode(src)
        pred = model.render(z, midi, vel, noise_seed=step)
        loss = mr_stft_loss(pred, tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % log_every == 0:
            rate = (step - start_step) / (time.time() - start)
            print(f"step {step:6d}  train {loss.item():.4f}  ({rate:.2f} steps/s)", flush=True)
        if step % eval_every == 0 or step == steps:
            val = validate(model, device, val_pairs)
            print(f"step {step:6d}  validation {val:.4f}", flush=True)
            if val < best:
                best = val
                torch.save({"model": model.state_dict(), "step": step,
                            "validation": val, "seed": seed}, out)
                print(f"           saved best -> {out}", flush=True)
            save_last(step)

    if start_step >= steps:
        print("already trained to the requested step count", flush=True)
    print(f"done. best validation {best:.4f}", flush=True)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # Measured on this Mac (MPS): batch 8 about 7.6 steps/s, batch 16 about 5.2 steps/s.
    p.add_argument("--steps", type=int, default=60000)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--resume", action="store_true",
                   help="continue from <out>_last.pt if it exists")
    args = p.parse_args(argv)
    train(args.steps, args.batch, args.lr, args.out, args.seed,
          args.log_every, args.eval_every, args.resume)


if __name__ == "__main__":
    main()
