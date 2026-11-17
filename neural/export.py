"""Command line: analyse a WAV sample and write a timbre file for the synth.

    .venv/bin/python -m neural.export sample.wav timbre.json
"""

import argparse
import sys
from pathlib import Path

from .analyze import analyze_file
from .timbre import write_timbre


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("wav", help="input sample (mono or stereo WAV)")
    parser.add_argument("out", help="output timbre JSON")
    args = parser.parse_args(argv)

    try:
        result = analyze_file(args.wav)
    except ValueError as err:
        print(f"error: {err}", file=sys.stderr)
        return 1

    write_timbre(args.out, result.harmonics, f0_hz=result.f0_hz,
                 source_name=Path(args.wav).name)
    audible = int((result.harmonics > 0.01).sum())
    print(f"f0 {result.f0_hz:.2f} Hz, {audible} audible harmonics -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
