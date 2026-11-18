"""Command line: analyse a WAV sample and write a timbre file for the synth.

    .venv/bin/python -m neural.export sample.wav timbre.json
"""

import argparse
import sys
from pathlib import Path

import librosa

from .analyze import analyze, analyze_file
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
    args = parser.parse_args(argv)

    try:
        if args.checkpoint:
            from .evaluate import load_model
            from .model import SR, export_from_model, fit_to_note
            model, _ = load_model(args.checkpoint)
            wave, _ = librosa.load(args.wav, sr=SR, mono=True)
            # Any sound is accepted here: the pitch is only metadata, so a
            # missing pitch is not an error on this path.
            harmonics, envelope = export_from_model(model, fit_to_note(wave))
            f0 = pitch_or_none(wave)
            method = f"model {args.checkpoint.name}"
        else:
            result = analyze_file(args.wav)
            harmonics, f0 = result.harmonics, result.f0_hz
            envelope = None
            method = "analysis"
    except ValueError as err:
        print(f"error: {err}", file=sys.stderr)
        return 1

    write_timbre(args.out, harmonics, f0_hz=f0, source_name=Path(args.wav).name,
                 envelope=envelope)
    audible = int((harmonics > 0.01).sum())
    shape = "with loudness envelope" if envelope is not None else "no envelope"
    pitch = "unpitched" if f0 is None else f"f0 {f0:.2f} Hz"
    print(f"[{method}] {pitch}, {audible} audible harmonics, {shape} -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
