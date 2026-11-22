"""Step 2d Step 1: where does the fitted patch's error remain?

For each note (valid split, one per instrument, family round-robin), fit the synth
controls with the shared fit, then split the log-spectral error of the baseline
(default controls) and of the fitted patch into regions:

  - harmonic versus off-harmonic bins (noise floor between partials)
  - attack (<0.1 s), sustain (0.1-3 s), release tail (>3 s)
  - low (<1 kHz), mid (1-4 kHz), high (>4 kHz) bands
  - per-harmonic level tracks over time (time-varying spectrum)
  - partial frequencies: how far each partial sits from k*f0, versus the source

Shares are the fraction of total absolute dB error in each region. The loss
shares tell us which tool to build first (see the plan, Step 3).

Run:  .venv/bin/python spikes/step2d_residual.py [--clips 24] [--seconds 30]
"""

import argparse
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
from scipy.signal import stft

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from neural.analyze import analyze  # noqa: E402
from neural.coreplay import Core, render_patch  # noqa: E402
from neural.fit import default_unit, fit_patch, unit_to_values  # noqa: E402
from neural.model import SR  # noqa: E402
from neural.movement import measure_movement  # noqa: E402
from step2d_fast_fit import select_notes  # noqa: E402

NFFT = 2048
HOP = 256
ATTACK_S = 0.1
SUSTAIN_END_S = 3.0
HARMONIC_WIDTH = 0.25   # bins within 25% of f0 of a partial count as harmonic
PARTIAL_WIDTH = 0.03    # search window for a partial's peak, fraction of k*f0
TRACK_HARMONICS = 6


def logspec(x):
    f, t, Z = stft(x, fs=SR, nperseg=NFFT, noverlap=NFFT - HOP, window="hann")
    return f, t, 20 * np.log10(np.abs(Z) + 1e-5)


def region_stats(render, target, f0):
    """Error shares by region, plus harmonic-track and partial-frequency measures."""
    n = min(len(render), len(target))
    f, t, Lr = logspec(render[:n])
    _, _, Lt = logspec(target[:n])
    E = np.abs(Lr - Lt)
    total = E.sum() + 1e-12
    out = {}

    if f0 is not None:
        harm = np.zeros(f.size, dtype=bool)
        for k in range(1, int(SR / 2 / f0) + 1):
            harm |= np.abs(f - k * f0) <= HARMONIC_WIDTH * f0
        out["share_off_harmonic"] = float(E[~harm].sum() / total)
        out["bins_off_harmonic"] = float((~harm).mean())

    attack = t < ATTACK_S
    sustain = (t >= ATTACK_S) & (t < SUSTAIN_END_S)
    release = t >= SUSTAIN_END_S
    for name, mask in (("attack", attack), ("sustain", sustain), ("release", release)):
        out[f"share_{name}"] = float(E[:, mask].sum() / total) if mask.any() else 0.0

    low, mid, high = f < 1000, (f >= 1000) & (f <= 4000), f > 4000
    for name, mask in (("low", low), ("mid", mid), ("high", high)):
        out[f"share_{name}"] = float(E[mask].sum() / total)

    if f0 is not None and sustain.any():
        track_err, static_err, dynamic_err = [], [], []
        for k in range(1, TRACK_HARMONICS + 1):
            if k * f0 >= SR / 2:
                break
            idx = int(np.argmin(np.abs(f - k * f0)))
            diff = Lr[idx, sustain] - Lt[idx, sustain]
            track_err.append(np.mean(np.abs(diff)))
            # Split: a fixed level offset (static) versus change over the note (dynamic).
            static_err.append(abs(np.mean(diff)))
            dynamic_err.append(np.mean(np.abs(diff - np.mean(diff))))
        out["track_db_error"] = float(np.mean(track_err)) if track_err else 0.0
        out["track_static_db"] = float(np.mean(static_err)) if static_err else 0.0
        out["track_dynamic_db"] = float(np.mean(dynamic_err)) if dynamic_err else 0.0

        mean_t = Lt[:, sustain].mean(axis=1)
        mean_r = Lr[:, sustain].mean(axis=1)
        cents_t, cents_err = [], []
        for k in range(2, 9):
            if k * f0 >= SR / 2:
                break
            window = np.where(np.abs(f - k * f0) <= PARTIAL_WIDTH * k * f0)[0]
            if window.size == 0:
                continue
            peak_t = window[np.argmax(mean_t[window])]
            if mean_t[peak_t] < mean_t.max() - 40:
                continue  # not a real partial in the source
            ct = 1200 * np.log2(f[peak_t] / (k * f0))
            peak_r = window[np.argmax(mean_r[window])]
            cr = 1200 * np.log2(f[peak_r] / (k * f0))
            cents_t.append(abs(ct))
            cents_err.append(abs(cr - ct))
        out["partial_cents_source"] = float(np.median(cents_t)) if cents_t else float("nan")
        out["partial_cents_error"] = float(np.median(cents_err)) if cents_err else float("nan")

    out["loss"] = float(E.mean())
    return out


def residual(args):
    index, path, pitch, velocity, wave, seconds = args
    try:
        analysis = analyze(wave, SR)
    except ValueError as exc:
        return {"index": index, "path": Path(path).name, "skipped": str(exc)}
    harmonics = analysis.harmonics
    movement = measure_movement(wave)
    f0 = analysis.f0_hz

    result = fit_patch(wave, harmonics, movement, midi=pitch, velocity=velocity,
                       seconds=seconds)
    hold_s = max(len(wave) / SR - 1.0, 0.5 * len(wave) / SR)
    tail_s = len(wave) / SR - hold_s
    core = Core(SR)
    core.set_timbre_moving(harmonics, movement)
    core.set_param("source_mix", 1.0)
    baseline = render_patch(core, unit_to_values(default_unit()), midi=pitch,
                            velocity=velocity, hold_s=hold_s, tail_s=tail_s)
    fitted = render_patch(core, result.patch, midi=pitch, velocity=velocity,
                          hold_s=hold_s, tail_s=tail_s)
    return {"index": index, "path": Path(path).name, "f0": f0,
            "baseline": region_stats(baseline, wave, f0),
            "fitted": region_stats(fitted, wave, f0)}


KEYS = ["loss", "share_off_harmonic", "bins_off_harmonic", "share_attack", "share_sustain",
        "share_release", "share_low", "share_mid", "share_high", "track_db_error",
        "track_static_db", "track_dynamic_db",
        "partial_cents_source", "partial_cents_error"]


def summarize(rows, label):
    print(f"\n{label} (median over {len(rows)} notes)")
    for key in KEYS:
        values = [r[label][key] for r in rows if key in r[label] and np.isfinite(r[label][key])]
        if values:
            print(f"  {key:>22}: {np.median(values):8.3f}   (IQR {np.percentile(values, 25):.3f}-"
                  f"{np.percentile(values, 75):.3f}, n={len(values)})")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--clips", type=int, default=24)
    parser.add_argument("--seconds", type=float, default=30.0)
    args = parser.parse_args(argv)

    notes = select_notes(args.clips * 2)
    jobs = [(i, p, pitch, vel, wave, args.seconds) for i, (p, pitch, vel, wave) in enumerate(notes)]
    with mp.get_context("spawn").Pool(4) as pool:
        outcomes = pool.map(residual, jobs)
    skipped = [r for r in outcomes if "skipped" in r]
    rows = sorted((r for r in outcomes if "skipped" not in r), key=lambda r: r["index"])[:args.clips]
    print(f"{len(rows)} notes analysed, {len(skipped)} skipped (no clear pitch)")
    summarize(rows, "baseline")
    summarize(rows, "fitted")
    print("\nRead the fitted shares: the largest one is the next thing to model.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
