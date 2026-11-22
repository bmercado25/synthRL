"""Test fixture for the full pipeline: a clip with a known tremolo, written as a WAV.

scripts/test.sh turns it into a version 3 timbre with neural.export, then the AU
test loads that file and checks that the plugin plays the movement.

    .venv/bin/python -m neural.fixtures build/pipeline/tremolo.wav
"""

import sys

import numpy as np

from .coreplay import write_wav
from .model import SR

TREMOLO_HZ = 6.0


def tremolo_clip(seconds=3.0, rate=TREMOLO_HZ):
    """A 220 Hz tone with a 6 Hz tremolo (depth 0.8), peak 0.3."""
    t = np.arange(int(SR * seconds)) / SR
    tone = sum(np.sin(2 * np.pi * 220 * k * t) / k for k in range(1, 5))
    return ((1.0 + 0.8 * np.sin(2 * np.pi * rate * t)) * tone * 0.3).astype(np.float32)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: python -m neural.fixtures OUT.wav", file=sys.stderr)
        return 2
    write_wav(argv[0], tremolo_clip(), SR)
    print(f"wrote {argv[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
