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

Version 4 adds an optional "patch" block (step 2d): synth control values from
the fit in neural/fit.py. It is a map from schema id to {"index", "value"}.
The index is the parameter's AU address, and the host applies each value
through the parameter tree after loading the timbre. Each value is checked
against the schema's range on write and on read.

Version 5 adds an optional "harmonics_attack" list (step 2d): the spectrum of
the first 100 ms, on the same scale as "harmonics". The synth blends toward it
after note-on, controlled by the Spectral Amount and Spectral Time parameters.
"""

import json
from pathlib import Path

import numpy as np

from .analyze import HARMONICS
from .coreplay import param_info
from .movement import MAX_FRAMES, Movement

FORMAT = "synth-timbre"
VERSION = 5
READABLE_VERSIONS = (1, 2, 3, 4, 5)
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


def _patch_doc(patch):
    """Validates a {schema id: value} patch and returns its file form."""
    doc = {}
    for identifier, value in patch.items():
        info = param_info(identifier)  # raises for an unknown id
        value = float(value)
        if not info["min"] <= value <= info["max"]:
            raise ValueError(f"{identifier} = {value} is outside "
                             f"{info['min']}..{info['max']}")
        doc[identifier] = {"index": int(info["index"]), "value": round(value, 6)}
    return doc


def read_attack(path):
    """The attack spectrum, or None for a file without one."""
    doc = json.loads(Path(path).read_text())
    block = doc.get("harmonics_attack")
    if block is None:
        return None
    attack = np.asarray(block, dtype=np.float32)
    if attack.shape != (HARMONICS,):
        raise ValueError(f"expected {HARMONICS} attack harmonics, got {attack.shape}")
    return attack


def read_patch(path):
    """The patch as {schema id: value}, or None for a file without one."""
    doc = json.loads(Path(path).read_text())
    block = doc.get("patch")
    if block is None:
        return None
    patch = {}
    for identifier, entry in block.items():
        info = param_info(identifier)
        value = float(entry["value"])
        if int(entry["index"]) != int(info["index"]):
            raise ValueError(f"{identifier}: index {entry['index']} does not match the schema")
        if not info["min"] <= value <= info["max"]:
            raise ValueError(f"{identifier} = {value} is outside {info['min']}..{info['max']}")
        patch[identifier] = value
    return patch


def write_timbre(path, harmonics, *, f0_hz, source_name, envelope=None, movement=None,
                 patch=None, attack_harmonics=None):
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
    if patch is not None:
        doc["patch"] = _patch_doc(patch)
    if attack_harmonics is not None:
        attack = np.asarray(attack_harmonics, dtype=np.float32)
        if attack.shape != (HARMONICS,):
            raise ValueError(f"expected {HARMONICS} attack harmonics, got {attack.shape}")
        doc["harmonics_attack"] = [round(float(h), 6) for h in attack]
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
