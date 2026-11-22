"""Step 2d ablation: which control groups help the fit, on the same notes.

Each configuration fits the same 24 validation notes (one per instrument, family
round-robin) with the same time cap and seeds. It reports the fit's reduction
against the fair neutral baseline, and a held-out spectral distance the fit never
sees. Results go to build/ablate/<name>.json so configurations can be compared.

Keep a group only if it passes the rule in the plan: median fitted loss at least
5% lower than the core fit, at least 60% of notes improved, and the held-out
distance also improves.

Run:  .venv/bin/python spikes/step2d_ablate.py --config core [--seconds 30]
"""

import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from neural.analyze import analyze  # noqa: E402
from neural.coreplay import Core, render_patch  # noqa: E402
from neural.fit import (CORE_CONTROLS, EXTRA_CONTROLS, FILTER_ENV_CONTROLS,  # noqa: E402
                        FILTER_MODE_CONTROLS, SPECTRAL_CONTROLS,
                        fit_patch, neutral_unit, unit_to_values)
from neural.model import SR  # noqa: E402
from neural.movement import measure_movement  # noqa: E402
from step2d_fast_fit import held_out_distance, select_notes  # noqa: E402

CONFIGS = {
    "core": CORE_CONTROLS,
    "filter_env": CORE_CONTROLS + FILTER_ENV_CONTROLS,
    "lfo_filter": CORE_CONTROLS + ["lfo_filter_depth"],
    "detune": CORE_CONTROLS + ["oscA_detune"],
    "reverb": CORE_CONTROLS + ["reverb_mix", "reverb_size"],
    "all_extra": CORE_CONTROLS + EXTRA_CONTROLS,
    "filter_mode": CORE_CONTROLS + FILTER_MODE_CONTROLS,
    "spectral": CORE_CONTROLS + SPECTRAL_CONTROLS,
}


def run_note(args):
    index, path, pitch, velocity, wave, seconds, controls = args
    try:
        analysis = analyze(wave, SR)
    except ValueError as exc:
        return {"index": index, "path": Path(path).name, "skipped": str(exc)}
    harmonics = analysis.harmonics
    movement = measure_movement(wave)
    # The attack spectrum is only used by configurations that free the spectral controls.
    attack = analysis.attack_harmonics if set(SPECTRAL_CONTROLS) & set(controls) else None
    result = fit_patch(wave, harmonics, movement, midi=pitch, velocity=velocity,
                       seconds=seconds, controls=controls, attack=attack)

    hold_s = max(len(wave) / SR - 1.0, 0.5 * len(wave) / SR)
    tail_s = len(wave) / SR - hold_s
    core = Core(SR)
    core.set_timbre_full(harmonics, attack, movement)
    core.set_param("source_mix", 1.0)
    neutral = render_patch(core, unit_to_values(neutral_unit(controls), controls),
                           midi=pitch, velocity=velocity, hold_s=hold_s, tail_s=tail_s)
    fitted = render_patch(core, result.patch, midi=pitch, velocity=velocity,
                          hold_s=hold_s, tail_s=tail_s)
    return {"index": index, "path": Path(path).name, "pitch": pitch,
            "neutral_fit": result.neutral_loss, "defaults_fit": result.defaults_loss,
            "fitted_fit": result.fitted_loss,
            "neutral_held": held_out_distance(neutral, wave),
            "fitted_held": held_out_distance(fitted, wave),
            "evals": result.evals, "moved": result.moved}


def reduction(a, b):
    return 100.0 * (1.0 - b / a)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", choices=sorted(CONFIGS), required=True)
    parser.add_argument("--clips", type=int, default=24)
    parser.add_argument("--split", default="valid", help="NSynth split: valid (dev) or test (final)")
    parser.add_argument("--seconds", type=float, default=30.0)
    args = parser.parse_args(argv)
    controls = CONFIGS[args.config]

    notes = select_notes(args.clips * 2, split=args.split)
    jobs = [(i, p, pitch, vel, wave, args.seconds, controls)
            for i, (p, pitch, vel, wave) in enumerate(notes)]
    with mp.get_context("spawn").Pool(4) as pool:
        outcomes = pool.map(run_note, jobs)
    rows = sorted((r for r in outcomes if "skipped" not in r), key=lambda r: r["index"])[:args.clips]

    fit_down = [reduction(r["neutral_fit"], r["fitted_fit"]) for r in rows]
    held_down = [reduction(r["neutral_held"], r["fitted_held"]) for r in rows]
    defaults_down = [reduction(r["defaults_fit"], r["fitted_fit"]) for r in rows]
    summary = {
        "config": args.config, "controls": controls, "notes": len(rows),
        "median_fit_reduction_vs_neutral": float(np.median(fit_down)),
        "median_fit_reduction_vs_defaults": float(np.median(defaults_down)),
        "notes_fit_improved": int(sum(d > 0 for d in fit_down)),
        "median_held_reduction": float(np.median(held_down)),
        "notes_held_improved": int(sum(d > 0 for d in held_down)),
        "median_evals": float(np.median([r["evals"] for r in rows])),
        "rows": rows,
    }
    out = ROOT / "build" / "ablate"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.config}_{args.split}.json").write_text(json.dumps(summary, indent=1, default=float))
    print(f"config {args.config} ({len(controls)} controls), {len(rows)} notes")
    print(f"  fit vs neutral: median {summary['median_fit_reduction_vs_neutral']:.1f}%, "
          f"improved {summary['notes_fit_improved']}/{len(rows)}")
    print(f"  fit vs defaults: median {summary['median_fit_reduction_vs_defaults']:.1f}%")
    print(f"  held-out vs neutral: median {summary['median_held_reduction']:.1f}%, "
          f"improved {summary['notes_held_improved']}/{len(rows)}")
    print(f"  median evaluations {summary['median_evals']:.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
