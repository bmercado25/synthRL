"""NSynth data for step 2b.

Splits (the official NSynth files, which have disjoint instruments):
  train      : NSynth "train" split, about 289k notes, used for training
  validation : NSynth "valid" split, used for checkpoint selection
  test       : NSynth "test" split, used only by neural.evaluate

NSynth notes are 4 s, 16 kHz, 16-bit mono, pitches MIDI 9-120, velocities 25-127.
Each note is identified by its name ("<instrument_str>-<pitch>-<velocity>").

Packing: reading about 290k small WAV files at random is slow, so
`python -m neural.data pack <split>` writes every note of a split into one
int16 array (`packed_int16.npy`, rows in examples.json order). The int16 values
are the WAV samples exactly, so packed and unpacked reads give the same audio.

Pairs: a training example is (source note, target note) from the same
instrument, with pitches and velocities that may differ. The model must rebuild
the target from the source, so the source must carry timbre, not pitch.
"""

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent / "data" / "nsynth"
SAMPLES = 64000  # 4 s at 16 kHz
PACKED_NAME = "packed_int16.npy"


@dataclass(frozen=True)
class Note:
    instrument: str
    pitch: int
    velocity: int
    path: str
    index: int = -1   # row in the packed split, or -1 if the split is not packed
    packed: str = ""  # path of the packed array for this note's split


def split_dir(split):
    return ROOT / split / f"nsynth-{split}"


def packed_path(split):
    return ROOT / split / PACKED_NAME


def load_split(split):
    base = split_dir(split)
    meta = json.loads((base / "examples.json").read_text())
    packed = packed_path(split)
    use_packed = packed.exists()
    notes = []
    for i, (key, x) in enumerate(meta.items()):
        notes.append(Note(
            x["instrument_str"], int(x["pitch"]), int(x["velocity"]),
            str(base / "audio" / f"{key}.wav"),
            index=i if use_packed else -1,
            packed=str(packed) if use_packed else ""))
    return notes


def pack_split(split):
    """Writes every note of a split into one int16 array, atomically."""
    base = split_dir(split)
    meta = json.loads((base / "examples.json").read_text())
    n = len(meta)
    out = packed_path(split)
    tmp = out.with_name(out.stem + ".tmp.npy")
    arr = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.int16, shape=(n, SAMPLES))
    for i, key in enumerate(meta):
        wave, sr = sf.read(str(base / "audio" / f"{key}.wav"), dtype="int16")
        if sr != 16000:
            raise ValueError(f"expected 16 kHz NSynth audio, got {sr}: {key}")
        if wave.ndim != 1:
            raise ValueError(f"expected mono NSynth audio: {key}")
        arr[i, :min(len(wave), SAMPLES)] = wave[:SAMPLES]
        if (i + 1) % 10000 == 0:
            print(f"  packed {i + 1}/{n}", flush=True)
    arr.flush()
    del arr
    tmp.rename(out)
    return out


_PACKED_CACHE = {}


def _packed(path):
    if path not in _PACKED_CACHE:
        _PACKED_CACHE[path] = np.load(path, mmap_mode="r")
    return _PACKED_CACHE[path]


def by_instrument(notes):
    groups = {}
    for n in notes:
        groups.setdefault(n.instrument, []).append(n)
    return groups


def read_wave(note):
    """Returns the note as float32 in [-1, 1), shape (SAMPLES,)."""
    if note.index >= 0:
        row = _packed(note.packed)[note.index]
        return row.astype(np.float32) / 32768.0
    wave, sr = sf.read(note.path, dtype="float32")
    if sr != 16000:
        raise ValueError(f"expected 16 kHz NSynth audio, got {sr}: {note.path}")
    return wave


class PairSampler:
    """Deterministic (source, target) pairs from the same instrument.

    Builds the instrument groups once, so sampling every training step is cheap.
    """

    def __init__(self, notes):
        groups = by_instrument(notes)
        self.groups = [groups[k] for k in sorted(groups) if len(groups[k]) >= 2]
        if not self.groups:
            raise ValueError("no instrument has two or more notes")

    def sample(self, count, seed):
        rng = random.Random(seed)
        pairs = []
        for _ in range(count):
            group = self.groups[rng.randrange(len(self.groups))]
            src, tgt = rng.sample(group, 2)
            pairs.append((src, tgt))
        return pairs


def sample_pairs(notes, count, seed):
    return PairSampler(notes).sample(count, seed)


def batch_from_pairs(pairs):
    """Loads pairs as arrays: sources (B,N), targets (B,N), midi/vel (B,)."""
    src = np.stack([read_wave(s) for s, _ in pairs])
    tgt = np.stack([read_wave(t) for _, t in pairs])
    midi = np.array([t.pitch for _, t in pairs], dtype=np.float32)
    vel = np.array([t.velocity for _, t in pairs], dtype=np.float32)
    return src, tgt, midi, vel


def main(argv=None):
    p = argparse.ArgumentParser(description="Pack an NSynth split into one array.")
    p.add_argument("command", choices=["pack"])
    p.add_argument("split", choices=["train", "valid", "test"])
    args = p.parse_args(argv)
    out = pack_split(args.split)
    print(f"packed {args.split} -> {out}")


if __name__ == "__main__":
    main()
