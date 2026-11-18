"""Timbre file format shared by the Python side and the plugin host.

JSON, so it is easy to inspect and the Swift host can read it with
JSONSerialization. The "harmonics" list is what the core consumes: element 0 is
the fundamental, element k-1 is harmonic k.
"""

import json
from pathlib import Path

import numpy as np

from .analyze import HARMONICS

FORMAT = "synth-timbre"
VERSION = 1


def write_timbre(path, harmonics, *, f0_hz, source_name):
    harmonics = np.asarray(harmonics, dtype=np.float32)
    if harmonics.shape != (HARMONICS,):
        raise ValueError(f"expected {HARMONICS} harmonics, got {harmonics.shape}")
    doc = {
        "format": FORMAT,
        "version": VERSION,
        "source": source_name,
        "f0_hz": round(float(f0_hz), 3),
        "harmonics": [round(float(h), 6) for h in harmonics],
    }
    Path(path).write_text(json.dumps(doc, indent=2) + "\n")


def read_timbre(path):
    doc = json.loads(Path(path).read_text())
    if doc.get("format") != FORMAT or doc.get("version") != VERSION:
        raise ValueError(f"not a {FORMAT} v{VERSION} file: {path}")
    harmonics = np.asarray(doc["harmonics"], dtype=np.float32)
    if harmonics.shape != (HARMONICS,):
        raise ValueError(f"expected {HARMONICS} harmonics, got {harmonics.shape}")
    return harmonics
