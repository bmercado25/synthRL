"""Timbre file format shared by the Python side and the plugin host.

JSON, so it is easy to inspect and the Swift host can read it with
JSONSerialization. The "harmonics" list is what the core consumes: element 0 is
the fundamental, element k-1 is harmonic k.

Version 2 adds an optional "envelope": loudness levels spread evenly over the
note, which the core applies to the neural source. Version 1 files (harmonics
only) are still read and play with a constant level.
"""

import json
from pathlib import Path

import numpy as np

from .analyze import HARMONICS

FORMAT = "synth-timbre"
VERSION = 2
READABLE_VERSIONS = (1, 2)
MAX_ENVELOPE = 64  # matches SYNTH_TIMBRE_MAX_ENV_POINTS in core/synth_core.h


def write_timbre(path, harmonics, *, f0_hz, source_name, envelope=None):
    harmonics = np.asarray(harmonics, dtype=np.float32)
    if harmonics.shape != (HARMONICS,):
        raise ValueError(f"expected {HARMONICS} harmonics, got {harmonics.shape}")
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
