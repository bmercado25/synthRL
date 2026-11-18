"""NSynth data for step 2b.

Splits:
  train      : NSynth "valid" split minus VAL_INSTRUMENTS held-out instruments
  validation : VAL_INSTRUMENTS instruments from the NSynth "valid" split
  test       : NSynth "test" split (its own instruments, never used in training)

NSynth notes are 4 s, 16 kHz, pitches MIDI 9-120, velocities 25-127.
Each note is identified by its name ("<instrument_str>-<pitch>-<velocity>").
The instrument name repeats across the valid and test splits, so splitting by
name would be wrong; the split is by official split file instead.

Pairs: a training example is (source note, target note) from the same
instrument, with pitches and velocities that may differ. The model must rebuild
the target from the source, so the source must carry timbre, not pitch.
"""

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent / "data" / "nsynth"
VAL_INSTRUMENTS = 8


@dataclass(frozen=True)
class Note:
    instrument: str
    pitch: int
    velocity: int
    path: str


def load_split(split):
    base = ROOT / split / f"nsynth-{split}"
    meta = json.loads((base / "examples.json").read_text())
    notes = []
    for key, x in meta.items():
        notes.append(Note(x["instrument_str"], int(x["pitch"]), int(x["velocity"]),
                          str(base / "audio" / f"{key}.wav")))
    return notes


def split_instruments(notes, seed=0):
    """Splits the valid split into train and validation by instrument name."""
    names = sorted({n.instrument for n in notes})
    rng = random.Random(seed)
    rng.shuffle(names)
    val = set(names[:VAL_INSTRUMENTS])
    train = [n for n in notes if n.instrument not in val]
    validation = [n for n in notes if n.instrument in val]
    return train, validation


def by_instrument(notes):
    groups = {}
    for n in notes:
        groups.setdefault(n.instrument, []).append(n)
    return groups


def read_wave(note):
    wave, sr = sf.read(note.path, dtype="float32")
    if sr != 16000:
        raise ValueError(f"expected 16 kHz NSynth audio, got {sr}: {note.path}")
    return wave


def sample_pairs(notes, count, seed):
    """Deterministic (source, target) pairs from the same instrument."""
    rng = random.Random(seed)
    groups = {k: v for k, v in by_instrument(notes).items() if len(v) >= 2}
    keys = sorted(groups)
    pairs = []
    for _ in range(count):
        key = keys[rng.randrange(len(keys))]
        src, tgt = rng.sample(groups[key], 2)
        pairs.append((src, tgt))
    return pairs


def batch_from_pairs(pairs):
    """Loads pairs as arrays: sources (B,N), targets (B,N), midi/vel (B,)."""
    src = np.stack([read_wave(s) for s, _ in pairs])
    tgt = np.stack([read_wave(t) for _, t in pairs])
    midi = np.array([t.pitch for _, t in pairs], dtype=np.float32)
    vel = np.array([t.velocity for _, t in pairs], dtype=np.float32)
    return src, tgt, midi, vel
