"""Command line: analyse a WAV sample and write a timbre file for the synth.

    .venv/bin/python -m neural.export sample.wav timbre.json
"""

import argparse
import sys
from pathlib import Path

import librosa

from .analyze import analyze, analyze_file
from .timbre import write_timbre


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
            from .model import SR, timbre_from_model
            model, _ = load_model(args.checkpoint)
            wave, _ = librosa.load(args.wav, sr=SR, mono=True)
            harmonics = timbre_from_model(model, wave)
            f0 = analyze(wave, SR).f0_hz
            method = f"model {args.checkpoint.name}"
        else:
            result = analyze_file(args.wav)
            harmonics, f0 = result.harmonics, result.f0_hz
            method = "analysis"
    except ValueError as err:
        print(f"error: {err}", file=sys.stderr)
        return 1

    write_timbre(args.out, harmonics, f0_hz=f0, source_name=Path(args.wav).name)
    audible = int((harmonics > 0.01).sum())
    print(f"[{method}] f0 {f0:.2f} Hz, {audible} audible harmonics -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
