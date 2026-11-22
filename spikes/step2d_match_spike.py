"""Step 2d Phase 0 spike: can the real core be fitted to a known patch?

Throwaway measurement, not the matcher. It answers three questions before the
matching engine is built:
  1. How fast is one evaluation (core render plus loss)?
  2. Is a render deterministic, in one process and across processes?
  3. Can a black-box optimizer recover a known patch from a perturbed start?

Patches are judged by loss, not by parameter error, because several parameters
trade off against each other (for example cutoff and filter envelope amount).

Run:  .venv/bin/python spikes/step2d_match_spike.py
"""

import hashlib
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from neural.coreplay import Core, render_patch  # noqa: E402
from neural.model import SR, mr_stft_loss  # noqa: E402

import json  # noqa: E402

SCHEMA = {p["id"]: p for p in json.loads((ROOT / "schema" / "synth_params.json").read_text())["parameters"]}

# Free parameters for the spike: amp envelope, filter, drive, filter envelope, LFO.
FREE = ["env_attack", "env_decay", "env_sustain", "env_release",
        "filter_cutoff", "filter_resonance", "drive",
        "filter_env_amount", "lfo_pitch_depth", "lfo_rate"]

HOLD_S = 1.0
TAIL_S = 0.5
HARMONICS = np.array([1.0 / k ** 1.2 for k in range(1, 17)] + [0.0] * 48, dtype=np.float32)


def to_values(unit):
    """0-1 vector -> schema values (linear between each parameter's min and max)."""
    out = {}
    for name, u in zip(FREE, unit):
        p = SCHEMA[name]
        out[name] = p["min"] + float(np.clip(u, 0.0, 1.0)) * (p["max"] - p["min"])
    return out


def envelope_db(x, rate=50):
    hop = SR // rate
    n = len(x) // hop
    rms = np.sqrt(np.mean(x[: n * hop].reshape(n, hop) ** 2, axis=1) + 1e-12)
    return np.maximum(20 * np.log10(rms), -60.0)


def loss(render, target):
    """Similarity loss: multi-resolution STFT on RMS-normalized audio, plus envelope in dB."""
    n = min(len(render), len(target))
    x = render[:n] / (np.sqrt(np.mean(render[:n] ** 2)) + 1e-9)
    y = target[:n] / (np.sqrt(np.mean(target[:n] ** 2)) + 1e-9)
    import torch
    spectral = float(mr_stft_loss(torch.as_tensor(x, dtype=torch.float32)[None],
                                  torch.as_tensor(y, dtype=torch.float32)[None]))
    env = float(np.mean(np.abs(envelope_db(render[:n]) - envelope_db(target[:n]))) / 20.0)
    return spectral + 0.5 * env


_CORE = None


def _core():
    global _CORE
    if _CORE is None:
        _CORE = Core(SR)
        _CORE.set_timbre(HARMONICS)
        _CORE.set_param("source_mix", 1.0)
    return _CORE


def render_unit(unit):
    return render_patch(_core(), to_values(unit), midi=60, velocity=110,
                        hold_s=HOLD_S, tail_s=TAIL_S)


def fingerprint(x):
    return hashlib.sha256(np.ascontiguousarray(x, dtype=np.float32).tobytes()).hexdigest()[:16]


def _child_render(unit):
    return fingerprint(render_unit(np.asarray(unit)))


def check_determinism(unit):
    """Same patch: identical render twice in-process, and identical in a fresh process."""
    a = fingerprint(render_unit(unit))
    b = fingerprint(render_unit(unit))
    with mp.get_context("spawn").Pool(1) as pool:
        c = pool.apply(_child_render, (list(unit),))
    return a == b, a == c


def measure_speed(unit, repeats=20):
    t0 = time.perf_counter()
    target = None
    for _ in range(repeats):
        target = render_unit(unit)
    render_s = (time.perf_counter() - t0) / repeats
    t0 = time.perf_counter()
    for _ in range(repeats):
        loss(target, target)
    loss_s = (time.perf_counter() - t0) / repeats
    return render_s, loss_s


class Budget:
    """Counts evaluations and returns the loss of a unit vector against a target."""

    def __init__(self, target):
        self.target = target
        self.evals = 0
        self.best = np.inf

    def __call__(self, unit):
        self.evals += 1
        value = loss(render_unit(np.asarray(unit)), self.target)
        self.best = min(self.best, value)
        return value


def recover(seed, budget_evals=400):
    """Known patch, perturbed start, scipy Powell within an evaluation budget."""
    from scipy.optimize import minimize

    rng = np.random.default_rng(seed)
    truth = rng.uniform(0.25, 0.75, size=len(FREE))
    target = render_unit(truth)
    start = np.clip(truth + rng.uniform(-0.2, 0.2, size=len(FREE)), 0, 1)
    f = Budget(target)
    initial = f(start)
    bounds = [(0.0, 1.0)] * len(FREE)
    minimize(f, start, method="Powell", bounds=bounds,
             options={"maxfev": budget_evals, "xtol": 1e-3, "ftol": 1e-4})
    return initial, f.best, f.evals


def main():
    unit = np.random.default_rng(0).uniform(0.3, 0.7, size=len(FREE))

    render_s, loss_s = measure_speed(unit)
    print(f"speed: render {render_s * 1000:.1f} ms per {HOLD_S + TAIL_S:.1f} s note, "
          f"loss {loss_s * 1000:.1f} ms, evaluation {(render_s + loss_s) * 1000:.1f} ms")
    print(f"        ~{1 / (render_s + loss_s):.0f} evaluations per second per process")

    same_process, cross_process = check_determinism(unit)
    print(f"determinism: same process {'PASS' if same_process else 'FAIL'}, "
          f"fresh process {'PASS' if cross_process else 'FAIL'}")

    wins = 0
    trials = 10
    for seed in range(trials):
        initial, best, evals = recover(seed)
        reduction = 1.0 - best / initial
        wins += reduction > 0.95
        print(f"recovery {seed}: loss {initial:.3f} -> {best:.4f} "
              f"({reduction * 100:.1f}% down, {evals} evals)")
    print(f"self-recovery: {wins}/{trials} with loss down over 95% "
          f"(proposed pass: 8 of 10)")


if __name__ == "__main__":
    main()
