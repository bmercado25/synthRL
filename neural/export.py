"""Command line: analyse a WAV sample and write a timbre file for the synth.

    .venv/bin/python -m neural.export sample.wav timbre.json
"""

import argparse
import sys
from pathlib import Path

import librosa

from .analyze import analyze, analyze_file
from .movement import measure_movement
from .timbre import write_timbre


def pitch_or_none(wave):
    """The clip's f0 in Hz, or None when it has no clear pitch."""
    from .model import SR
    try:
        return analyze(wave, SR).f0_hz
    except ValueError:
        return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("wav", help="input sample (mono or stereo WAV)")
    parser.add_argument("out", help="output timbre JSON")
    parser.add_argument("--checkpoint", type=Path,
                        help="use a trained step 2b model instead of the step 2a analysis")
    parser.add_argument("--no-movement", action="store_true",
                        help="leave out the loudness, pitch and brightness movement (step 2c)")
    parser.add_argument("--fit", action="store_true",
                        help="also fit the synth controls to the note (step 2d); "
                             "needs build/libsynthcore.dylib")
    parser.add_argument("--fit-seconds", type=float, default=30.0,
                        help="time cap for the fit (default 30)")
    args = parser.parse_args(argv)

    from .model import SR
    try:
        wave, _ = librosa.load(args.wav, sr=SR, mono=True)
        movement = None
        attack = None  # only the analysis path measures an attack spectrum
        if args.checkpoint:
            from .evaluate import load_model
            from .model import export_from_model, fit_to_note
            model, _ = load_model(args.checkpoint)
            # Any sound is accepted here: the pitch is only metadata, so a
            # missing pitch is not an error on this path.
            harmonics, envelope = export_from_model(model, fit_to_note(wave))
            f0 = pitch_or_none(wave)
            method = f"model {args.checkpoint.name}"
        else:
            result = analyze_file(args.wav)
            harmonics, f0 = result.harmonics, result.f0_hz
            attack = result.attack_harmonics
            envelope = None
            method = "analysis"
        if not args.no_movement:
            # Movement replaces the decoder's envelope, so the envelope is dropped.
            movement = measure_movement(wave)
            envelope = None
    except ValueError as err:
        print(f"error: {err}", file=sys.stderr)
        return 1

    patch = None
    if args.fit:
        from .fit import fit_patch
        import numpy as np
        midi = 60 if f0 is None else int(round(69 + 12 * np.log2(f0 / 440.0)))
        try:
            result = fit_patch(wave, harmonics, movement, midi=midi,
                               seconds=args.fit_seconds)
        except (OSError, ValueError) as err:
            print(f"error: fit failed: {err}", file=sys.stderr)
            return 1
        patch = result.patch
        print(f"fit at MIDI {midi}: loss {result.neutral_loss:.3f} -> "
              f"{result.fitted_loss:.3f} in {result.seconds:.1f} s, {result.evals} evaluations")

    write_timbre(args.out, harmonics, f0_hz=f0, source_name=Path(args.wav).name,
                 envelope=envelope, movement=movement, patch=patch,
                 attack_harmonics=attack)
    audible = int((harmonics > 0.01).sum())
    if movement is not None:
        shape = (f"movement {len(movement) / movement.rate_hz:.2f} s, "
                 f"loop {movement.loop_start}-{movement.loop_end} frames")
    elif envelope is not None:
        shape = "with loudness envelope"
    else:
        shape = "no envelope"
    pitch = "unpitched" if f0 is None else f"f0 {f0:.2f} Hz"
    print(f"[{method}] {pitch}, {audible} audible harmonics, {shape} -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
