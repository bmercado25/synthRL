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
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

FREE = ["env_attack", "env_decay", "env_sustain", "env_release",
        "filter_cutoff", "filter_resonance", "drive",
        "filter_env_amount", "lfo_pitch_depth", "lfo_rate"]

# Staged CMA-ES, as in the Phase 0b diagnostics: groups first, then everything.
STAGES = [
    (["env_attack", "env_decay", "env_sustain", "env_release", "filter_cutoff"], 0.4),
    (["filter_env_amount", "lfo_pitch_depth", "lfo_rate"], 0.4),
    (["filter_resonance", "drive"], 0.3),
    (FREE, 0.2),
]

SCHEMA = {p["id"]: p for p in json.loads((ROOT / "schema" / "synth_params.json").read_text())["parameters"]}
HOLD_S = 3.0  # NSynth notes are 3 s held, then a release tail to 4 s


def select_notes(count):
    """Pitch-60 notes, one per instrument, round-robin across instrument families."""
    from neural.data import load_split, read_wave
    by_family = {}
    for note in sorted(load_split("valid"), key=lambda n: (n.instrument, -n.velocity)):
        if note.pitch != 60:
            continue
        family = note.instrument.split("_")[0]
        group = by_family.setdefault(family, [])
        if not group or group[-1].instrument != note.instrument:
            group.append(note)
    queues = [list(v) for _, v in sorted(by_family.items())]
    chosen = []
    while queues and len(chosen) < count:
        for queue in queues:
            if queue and len(chosen) < count:
                note = queue.pop(0)
                chosen.append((note.path, note.pitch, note.velocity, read_wave(note)))
    return chosen


def prepare(wave):
    """Analysis and movement for one note, at its own pitch."""
    from neural.analyze import analyze
    from neural.model import SR
    from neural.movement import measure_movement
    analysis = analyze(wave, SR)
    midi = int(round(69 + 12 * np.log2(analysis.f0_hz / 440.0)))
    return analysis.harmonics, measure_movement(wave), midi


def unit_to_values(unit):
    out = {}
    for name, u in zip(FREE, unit):
        p = SCHEMA[name]
        out[name] = p["min"] + float(np.clip(u, 0.0, 1.0)) * (p["max"] - p["min"])
    return out


def default_unit():
    return np.array([(SCHEMA[n]["default"] - SCHEMA[n]["min"]) /
                     (SCHEMA[n]["max"] - SCHEMA[n]["min"]) for n in FREE])


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
    """Worker: baseline, time-capped staged CMA-ES, held-out check for one note."""
    import cma
    from neural.coreplay import Core, render_patch
    from neural.model import SR
    import step2d_match_spike as spike

    index, path, pitch, velocity, wave, seconds = args
    try:
        harmonics, movement, midi = prepare(wave)
    except ValueError as exc:  # no clear pitch: the plugin would fall back to note 60
        return {"index": index, "path": Path(path).name, "skipped": str(exc)}
    target = wave
    tail_s = len(target) / SR - HOLD_S

    core = Core(SR)
    core.set_timbre_moving(harmonics, movement)
    core.set_param("source_mix", 1.0)

    def render(unit):
        return render_patch(core, unit_to_values(unit), midi=midi, velocity=velocity,
                            hold_s=HOLD_S, tail_s=tail_s)

    base = default_unit()
    baseline_render = render(base)
    baseline_fit = spike.loss(baseline_render, target)
    baseline_held = held_out_distance(baseline_render, target)

    best = {"x": base.copy(), "f": baseline_fit, "evals": 0}
    deadline = time.perf_counter() + seconds
    t0 = time.perf_counter()

    def objective(unit):
        unit = np.clip(unit, 0.0, 1.0)
        if time.perf_counter() > deadline:
            raise TimeoutError
        best["evals"] += 1
        value = spike.loss(render(unit), target)
        if value < best["f"]:
            best["f"], best["x"] = value, unit.copy()
        return value

    x = base.copy()
    try:
        for stage, (names, sigma) in enumerate(STAGES):
            idx = [FREE.index(n) for n in names]

            def sub(z, idx=idx):
                y = best["x"].copy()
                y[idx] = np.clip(z, 0.0, 1.0)
                return objective(y)

            es = cma.CMAEvolutionStrategy(best["x"][idx], sigma,
                                          {"bounds": [0.0, 1.0], "verbose": -9,
                                           "seed": index * 10 + stage, "maxfevals": 10 ** 6,
                                           "tolfun": 1e-12, "tolx": 1e-7})
            while not es.stop():
                cands = es.ask()
                es.tell(cands, [sub(z) for z in cands])
    except TimeoutError:
        pass
    elapsed = time.perf_counter() - t0

    fitted_render = render(best["x"])
    fitted_held = held_out_distance(fitted_render, target)
    moved = {n: round(float(best["x"][i]), 3) for i, n in enumerate(FREE)
             if abs(best["x"][i] - base[i]) > 0.05}
    return {"index": index, "path": Path(path).name, "pitch": pitch, "midi": midi,
            "baseline_fit": baseline_fit, "fitted_fit": best["f"],
            "baseline_held": baseline_held, "fitted_held": fitted_held,
            "evals": best["evals"], "seconds": elapsed, "moved": moved,
            "patch": {n: round(unit_to_values(best["x"])[n], 4) for n in FREE}}


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
    print(f"\n{'note':<34} {'midi':>4} {'fit base':>8} {'fit':>7} {'down%':>6} "
          f"{'held base':>9} {'held':>6} {'down%':>6} {'evals':>6}")
    for r in results:
        fit_down = 100 * (1 - r["fitted_fit"] / r["baseline_fit"])
        held_down = 100 * (1 - r["fitted_held"] / r["baseline_held"])
        print(f"{r['path']:<34} {r['midi']:>4} {r['baseline_fit']:8.3f} {r['fitted_fit']:7.3f} "
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
        print(f"  {r['path']} (MIDI {r['midi']}):")
        for name, value in r["patch"].items():
            print(f"    {name} = {value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
