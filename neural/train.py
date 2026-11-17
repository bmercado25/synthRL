"""Step 2b training: cross-pitch reconstruction on NSynth.

    .venv/bin/python -m neural.train --steps 4000

Each step samples (source, target) notes from one training instrument and
trains the model to rebuild the target from the source, at the target's pitch
and velocity. Validation uses fixed pairs from held-out training instruments.
The best checkpoint by validation loss is kept.
"""

import argparse
import time
from pathlib import Path

import torch

from . import data
from .model import NeuralSource, mr_stft_loss

DEFAULT_OUT = data.ROOT.parent / "checkpoints" / "nsynth_2b.pt"


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


def train(steps, batch, lr, out, seed, log_every, eval_every):
    torch.manual_seed(seed)
    device = pick_device()
    train_notes, val_notes = data.split_instruments(data.load_split("valid"), seed=seed)
    val_pairs = data.sample_pairs(val_notes, count=128, seed=1234)
    print(f"device {device}: {len(train_notes)} train notes, {len(val_notes)} validation notes, "
          f"{len({n.instrument for n in train_notes})} train instruments", flush=True)

    model = NeuralSource().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    best = float("inf")
    out.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()

    for step in range(1, steps + 1):
        pairs = data.sample_pairs(train_notes, count=batch, seed=seed * 1_000_003 + step)
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
            rate = step / (time.time() - start)
            print(f"step {step:5d}  train {loss.item():.4f}  ({rate:.2f} steps/s)", flush=True)
        if step % eval_every == 0 or step == steps:
            val = validate(model, device, val_pairs)
            print(f"step {step:5d}  validation {val:.4f}", flush=True)
            if val < best:
                best = val
                torch.save({"model": model.state_dict(), "step": step,
                            "validation": val, "seed": seed}, out)
                print(f"           saved best -> {out}", flush=True)

    print(f"done. best validation {best:.4f}", flush=True)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=500)
    args = p.parse_args(argv)
    train(args.steps, args.batch, args.lr, args.out, args.seed, args.log_every, args.eval_every)


if __name__ == "__main__":
    main()
