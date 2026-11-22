"""Step 2d practical check: does a fast fit on real NSynth notes beat the baseline?

The baseline is the current step 2c render: the analysed harmonic timbre with its
measured movement and default synth controls. The fit adjusts the synth controls
only (amplitude envelope, filter, LFO, drive) with staged CMA-ES, under a time
cap per note. The fit is judged with two measures:
  - the loss the optimizer uses (fit loss);
  - a held-out log-spectral distance on a 4096-point STFT, which the optimizer
    never sees, to guard against optimizing the metric.

Each note is rendered at its own pitch (from pYIN) and with its own movement, so
the baseline is what the plugin plays today for that sample.

Run:  .venv/bin/python spikes/step2d_fast_fit.py [--clips 8] [--seconds 30]
"""

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

def select_notes(count, split="valid"):
    """Pitch-60 notes, one per instrument, round-robin across instrument families."""
    from neural.data import load_split, read_wave
    by_family = {}
    for note in sorted(load_split(split), key=lambda n: (n.instrument, -n.velocity)):
        if note.pitch != 60:
            continue
        family = note.instrument.split("_")[0]
        group = by_family.setdefault(family, [])
        if not group or group[-1].instrument != note.instrument:
            group.append(note)
    queues = [list(v) for _, v in sorted(by_family.items())]
    chosen = []
    while len(chosen) < count:
        added = False
        for queue in queues:
            if queue and len(chosen) < count:
                note = queue.pop(0)
                chosen.append((note.path, note.pitch, note.velocity, read_wave(note)))
                added = True
        if not added:  # every family is exhausted: fewer notes than requested
            break
    return chosen


def held_out_distance(render, target):
    """Log-spectral L1 on a 4096-point STFT, in dB. Not used by the fit."""
    import torch
    n = min(len(render), len(target))
    window = torch.hann_window(4096)
    def logmag(x):
        spec = torch.stft(torch.as_tensor(x[:n], dtype=torch.float32), n_fft=4096,
                          hop_length=1024, window=window, return_complex=True).abs()
        return 20 * torch.log10(spec + 1e-5)
    return float((logmag(render) - logmag(target)).abs().mean())


def fit_note(args):
    """Worker: one note, fitted with the shared fit and checked on a held-out measure."""
    from neural.coreplay import Core, render_patch
    from neural.fit import fit_patch, default_unit, unit_to_values
    from neural.analyze import analyze
    from neural.model import SR
    from neural.movement import measure_movement

    index, path, pitch, velocity, wave, seconds = args
    try:
        analysis = analyze(wave, SR)
    except ValueError as exc:  # no clear pitch: the plugin would fall back to note 60
        return {"index": index, "path": Path(path).name, "skipped": str(exc)}
    harmonics = analysis.harmonics
    movement = measure_movement(wave)
    # The NSynth label is ground truth here, so an octave error in pYIN cannot
    # move the fit to the wrong pitch. Its analysed pitch is reported as a check.
    analysed_midi = int(round(69 + 12 * np.log2(analysis.f0_hz / 440.0)))

    result = fit_patch(wave, harmonics, movement, midi=pitch, velocity=velocity,
                       seconds=seconds)
    hold_s = max(len(wave) / SR - 1.0, 0.5 * len(wave) / SR)
    core = Core(SR)
    core.set_timbre_moving(harmonics, movement)
    core.set_param("source_mix", 1.0)
    baseline = render_patch(core, unit_to_values(default_unit()), midi=pitch,
                            velocity=velocity, hold_s=hold_s, tail_s=len(wave) / SR - hold_s)
    fitted = render_patch(core, result.patch, midi=pitch, velocity=velocity,
                          hold_s=hold_s, tail_s=len(wave) / SR - hold_s)
    return {"index": index, "path": Path(path).name, "pitch": pitch, "midi": analysed_midi,
            "baseline_fit": result.neutral_loss, "fitted_fit": result.fitted_loss,
            "baseline_held": held_out_distance(baseline, wave),
            "fitted_held": held_out_distance(fitted, wave),
            "evals": result.evals, "seconds": result.seconds,
            "moved": {n: result.patch[n] for n in result.moved}, "patch": result.patch}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--clips", type=int, default=8)
    parser.add_argument("--seconds", type=float, default=30.0)
    args = parser.parse_args(argv)

    # Over-select, because some notes have no clear pitch and are skipped.
    notes = select_notes(args.clips * 2)
    print(f"{len(notes)} candidate notes from distinct instruments (validation split)")
    jobs = [(i, p, pitch, vel, wave, args.seconds) for i, (p, pitch, vel, wave) in enumerate(notes)]
    with mp.get_context("spawn").Pool(4) as pool:
        outcomes = pool.map(fit_note, jobs)

    skipped = [r for r in outcomes if "skipped" in r]
    for r in skipped:
        print(f"skipped {r['path']}: {r['skipped']}")
    results = sorted((r for r in outcomes if "skipped" not in r), key=lambda r: r["index"])[:args.clips]
    print(f"{len(results)} fitted, {len(skipped)} skipped (no clear pitch)")
    print(f"\n{'note':<34} {'label':>5} {'pyin':>4} {'fit base':>8} {'fit':>7} {'down%':>6} "
          f"{'held base':>9} {'held':>6} {'down%':>6} {'evals':>6}")
    for r in results:
        fit_down = 100 * (1 - r["fitted_fit"] / r["baseline_fit"])
        held_down = 100 * (1 - r["fitted_held"] / r["baseline_held"])
        print(f"{r['path']:<34} {r['pitch']:>5} {r['midi']:>4} {r['baseline_fit']:8.3f} {r['fitted_fit']:7.3f} "
              f"{fit_down:6.1f} {r['baseline_held']:9.2f} {r['fitted_held']:6.2f} "
              f"{held_down:6.1f} {r['evals']:6d}")
    fit_medians = [100 * (1 - r["fitted_fit"] / r["baseline_fit"]) for r in results]
    held_medians = [100 * (1 - r["fitted_held"] / r["baseline_held"]) for r in results]
    improved = sum(h > 0 for h in held_medians)
    print(f"\nmedian fit loss reduction {np.median(fit_medians):.1f}%, "
          f"median held-out reduction {np.median(held_medians):.1f}%, "
          f"held-out improved on {improved}/{len(results)} notes")
    for r in results:
        print(f"  {r['path']}: moved {r['moved']}")
    print("\nFitted patch, to set by hand in the app (schema values):")
    for r in results:
        print(f"  {r['path']} (MIDI {r['pitch']}):")
        for name, value in r["patch"].items():
            print(f"    {name} = {value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
