"""Step 2b evaluation on held-out instruments (the NSynth test split).

    .venv/bin/python -m neural.evaluate --checkpoint data/checkpoints/nsynth_2b.pt

Each pair is (source note, target note) from one held-out instrument. Every
method builds the target's pitch and velocity from the source only:

  model_full   the decoder at the target pitch and velocity (training objective)
  model_timbre the pitch-independent timbre the plugin would load, rendered
               at the target pitch with the reference renderer
  analysis_2a  the step 2a analysis timbre, rendered the same way
  mean_timbre  the average 2a timbre over training instruments (no source info)
  repitch      the source note shifted to the target pitch (no model)

Metrics are computed on peak-normalised audio so loudness does not decide the
result. Pitch is checked with pYIN on the output.
"""

import argparse
import json
import math
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

from . import data
from .analyze import analyze
from .model import (NOTE_SECONDS, SR, NeuralSource, mr_stft_loss,
                    timbre_from_model)
from .render import render_note

OUT_DIR = Path(__file__).resolve().parent.parent / "build" / "neural"
N_SAMPLES = int(SR * NOTE_SECONDS)


def peak_norm(x):
    p = float(np.max(np.abs(x)))
    return x / p if p > 0 else x


def loss_on(pred, tgt):
    a = torch.as_tensor(peak_norm(pred), dtype=torch.float32)[None]
    b = torch.as_tensor(peak_norm(tgt), dtype=torch.float32)[None]
    return float(mr_stft_loss(a, b))


def pitch_error_cents(wave, target_midi):
    hz = librosa.midi_to_hz(target_midi)
    # A search band of one octave each side. A narrow band (a semitone or so)
    # makes librosa's pYIN fail internally, so the band is kept wide.
    # The upper edge is capped below the 8 kHz Nyquist limit, which high notes exceed.
    f0, _, prob = librosa.pyin(wave, fmin=hz / 2, fmax=min(hz * 2, 0.95 * SR / 2),
                               sr=SR, frame_length=2048, hop_length=512)
    good = np.isfinite(f0) & (prob >= 0.5)
    if not good.any():
        return float("nan")
    return float(1200 * math.log2(np.median(f0[good]) / hz))


def load_model(path):
    ckpt = torch.load(path, map_location="cpu")
    model = NeuralSource()
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def mean_training_timbre(train_notes, count=100, seed=7):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(train_notes), size=min(count, len(train_notes)), replace=False)
    profiles = []
    for i in idx:
        note = train_notes[i]
        try:
            profiles.append(analyze(data.read_wave(note), SR).harmonics)
        except ValueError:
            continue
    mean = np.mean(profiles, axis=0)
    return (mean / mean.max()).astype(np.float32)


def method_outputs(model, mean_profile, src_wave, src_note, tgt):
    """Returns {method: wave}. analysis_2a is left out if the source has no clear pitch."""
    midi = torch.tensor([float(tgt.pitch)])
    vel = torch.tensor([float(tgt.velocity)])
    with torch.no_grad():
        z = model.encode(torch.as_tensor(src_wave, dtype=torch.float32)[None])
        full = model.render(z, midi, vel, noise_seed=0)[0].numpy()
    timbre = timbre_from_model(model, src_wave)
    shifted = librosa.effects.pitch_shift(src_wave, sr=SR, n_steps=tgt.pitch - src_note.pitch)
    shifted = np.pad(shifted, (0, max(0, N_SAMPLES - len(shifted))))[:N_SAMPLES]
    out = {
        "model_full": full,
        "model_timbre": render_note(timbre, tgt.pitch, tgt.velocity, NOTE_SECONDS, SR),
        "mean_timbre": render_note(mean_profile, tgt.pitch, tgt.velocity, NOTE_SECONDS, SR),
        "repitch": shifted,
    }
    try:
        analysis = analyze(src_wave, SR).harmonics
        out["analysis_2a"] = render_note(analysis, tgt.pitch, tgt.velocity, NOTE_SECONDS, SR)
    except ValueError:
        pass
    return out


def summarise(values):
    v = np.array(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0}
    return {"n": int(v.size), "mean": float(v.mean()),
            "median": float(np.median(v)),
            "sem": float(v.std(ddof=1) / math.sqrt(v.size)) if v.size > 1 else 0.0}


def velocity_check(model, src_wave):
    vels = [25, 50, 75, 100, 127]
    rms = []
    with torch.no_grad():
        z = model.encode(torch.as_tensor(src_wave, dtype=torch.float32)[None])
        for v in vels:
            y = model.render(z, torch.tensor([60.0]), torch.tensor([float(v)]), noise_seed=0)[0]
            rms.append(float(torch.sqrt((y ** 2).mean())))
    monotonic = all(b > a for a, b in zip(rms, rms[1:]))
    return {"velocities": vels, "rms": rms, "monotonic": monotonic}


def speed_check(model, src_wave, repeats=3):
    z_src = torch.as_tensor(src_wave, dtype=torch.float32)[None]
    with torch.no_grad():
        z = model.encode(z_src)
        model.render(z, torch.tensor([60.0]), torch.tensor([100.0]))
        best = float("inf")
        for _ in range(repeats):
            start = time.perf_counter()
            model.render(z, torch.tensor([60.0]), torch.tensor([100.0]))
            best = min(best, time.perf_counter() - start)
    return {"device": "cpu", "seconds_per_4s_note": best, "realtime_factor": best / NOTE_SECONDS}


def determinism_check(model, src_wave):
    with torch.no_grad():
        z = model.encode(torch.as_tensor(src_wave, dtype=torch.float32)[None])
        a = model.render(z, torch.tensor([60.0]), torch.tensor([100.0]), noise_seed=5)
        b = model.render(z, torch.tensor([60.0]), torch.tensor([100.0]), noise_seed=5)
    return bool(torch.equal(a, b))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--pairs", type=int, default=200)
    p.add_argument("--pitch-pairs", type=int, default=60,
                   help="pitch is checked with pYIN, which is slow, so use a subset")
    p.add_argument("--listen", type=int, default=2, help="pairs to write as WAV for listening")
    args = p.parse_args(argv)

    torch.set_num_threads(max(1, torch.get_num_threads()))
    model, ckpt = load_model(args.checkpoint)
    train_notes, _ = data.split_instruments(data.load_split("valid"), seed=ckpt.get("seed", 0))
    test_notes = data.load_split("test")
    pairs = data.sample_pairs(test_notes, count=args.pairs, seed=99)
    print(f"checkpoint step {ckpt['step']} (validation {ckpt['validation']:.4f}); "
          f"{len(pairs)} held-out pairs", flush=True)

    mean_profile = mean_training_timbre(train_notes)
    results = {m: {"loss": [], "pitch": []} for m in
               ["model_full", "model_timbre", "analysis_2a", "mean_timbre", "repitch"]}
    failures = 0
    listen_dir = OUT_DIR / "listen"
    listen_dir.mkdir(parents=True, exist_ok=True)

    for i, (src_note, tgt_note) in enumerate(pairs):
        src_wave = data.read_wave(src_note)
        tgt_wave = data.read_wave(tgt_note)
        outputs = method_outputs(model, mean_profile, src_wave, src_note, tgt_note)
        if "analysis_2a" not in outputs:
            failures += 1  # only the analysis baseline is missing for this pair
        check_pitch = i < args.pitch_pairs
        for name, wave in outputs.items():
            results[name]["loss"].append(loss_on(wave, tgt_wave))
            if check_pitch:
                results[name]["pitch"].append(abs(pitch_error_cents(wave, tgt_note.pitch)))
        if i < args.listen:
            for label, wave in [("source", src_wave), ("target", tgt_wave), *outputs.items()]:
                sf.write(listen_dir / f"pair{i}_{label}.wav", peak_norm(wave), SR)
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(pairs)} pairs", flush=True)

    summary = {m: {"stft_loss": summarise(v["loss"]), "pitch_abs_cents": summarise(v["pitch"])}
               for m, v in results.items()}
    first_src = data.read_wave(pairs[0][0])
    extra = {
        "velocity": velocity_check(model, first_src),
        "speed": speed_check(model, first_src),
        "deterministic": determinism_check(model, first_src),
        "analysis_failures_skipped": failures,
    }
    report = {"checkpoint_step": ckpt["step"], "pairs": len(pairs), "methods": summary, **extra}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "eval.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
