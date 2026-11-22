"""Timbre file format shared by the Python side and the plugin host.

JSON, so it is easy to inspect and the Swift host can read it with
JSONSerialization. The "harmonics" list is what the core consumes: element 0 is
the fundamental, element k-1 is harmonic k.

Version 2 adds an optional "envelope": loudness levels spread evenly over the
note, which the core applies to the neural source. Version 1 files (harmonics
only) are still read and play with a constant level.

Version 3 adds an optional "movement" block (step 2c): the source's loudness,
pitch and brightness tracks on a 50 Hz grid, plus a loop region. When it is
present the core plays it instead of the envelope, scaled by the Movement
control. Versions 1 and 2 are still read.
"""

import json
from pathlib import Path

import numpy as np

from .analyze import HARMONICS
from .movement import MAX_FRAMES, Movement

FORMAT = "synth-timbre"
VERSION = 3
READABLE_VERSIONS = (1, 2, 3)
MAX_ENVELOPE = 64  # matches SYNTH_TIMBRE_MAX_ENV_POINTS in core/synth_core.h


def _movement_doc(movement):
    if len(movement) > MAX_FRAMES:
        raise ValueError(f"movement has more than {MAX_FRAMES} frames")
    return {
        "rate_hz": float(movement.rate_hz),
        "loop_start": int(movement.loop_start),
        "loop_end": int(movement.loop_end),
        "amp": [round(float(v), 6) for v in movement.amp],
        "pitch_cents": [round(float(v), 3) for v in movement.pitch_cents],
        "brightness_oct": [round(float(v), 6) for v in movement.brightness_oct],
    }


def write_timbre(path, harmonics, *, f0_hz, source_name, envelope=None, movement=None):
    harmonics = np.asarray(harmonics, dtype=np.float32)
    if harmonics.shape != (HARMONICS,):
        raise ValueError(f"expected {HARMONICS} harmonics, got {harmonics.shape}")
    if envelope is not None and movement is not None:
        raise ValueError("a timbre has an envelope or movement, not both")
    doc = {
        "format": FORMAT,
        "version": VERSION,
        "source": source_name,
        # None means the source had no clear pitch (Model mode). Readers ignore f0.
        "f0_hz": None if f0_hz is None else round(float(f0_hz), 3),
        "harmonics": [round(float(h), 6) for h in harmonics],
    }
    if envelope is not None:
        env = np.asarray(envelope, dtype=np.float32)
        if env.ndim != 1 or not 1 <= env.size <= MAX_ENVELOPE:
            raise ValueError(f"envelope must have 1 to {MAX_ENVELOPE} points")
        doc["envelope"] = [round(float(e), 6) for e in env]
    if movement is not None:
        doc["movement"] = _movement_doc(movement)
    Path(path).write_text(json.dumps(doc, indent=2) + "\n")


def read_timbre(path):
    doc = json.loads(Path(path).read_text())
    if doc.get("format") != FORMAT or doc.get("version") not in READABLE_VERSIONS:
        raise ValueError(f"not a {FORMAT} file: {path}")
    harmonics = np.asarray(doc["harmonics"], dtype=np.float32)
    if harmonics.shape != (HARMONICS,):
        raise ValueError(f"expected {HARMONICS} harmonics, got {harmonics.shape}")
    return harmonics


def read_envelope(path):
    """The loudness envelope, or None for a file without one."""
    doc = json.loads(Path(path).read_text())
    if "envelope" not in doc:
        return None
    return np.asarray(doc["envelope"], dtype=np.float32)


def read_movement(path):
    """The movement tracks as a Movement, or None for a file without them."""
    doc = json.loads(Path(path).read_text())
    block = doc.get("movement")
    if block is None:
        return None
    return Movement(
        rate_hz=float(block["rate_hz"]),
        loop_start=int(block["loop_start"]),
        loop_end=int(block["loop_end"]),
        amp=np.asarray(block["amp"], dtype=np.float32),
        pitch_cents=np.asarray(block["pitch_cents"], dtype=np.float32),
        brightness_oct=np.asarray(block["brightness_oct"], dtype=np.float32),
    )
