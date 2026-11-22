"""Drives the real C++ core from Python (ctypes), for evaluation and listening WAVs.

The renders match what the plugin plays, because it is the same core
(core/synth_core.cpp) built as build/libsynthcore.dylib by scripts/test.sh.
Parameter indices come from schema/synth_params.json, so nothing is hand-numbered.

    from neural.coreplay import render_note
    audio = render_note(harmonics, movement=track, seconds=8.0)
"""

import ctypes
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LIB_PATH = ROOT / "build" / "libsynthcore.dylib"
SCHEMA_PATH = ROOT / "schema" / "synth_params.json"

MAX_HARMONICS = 64
MAX_FRAMES = 500


class _SynthMovement(ctypes.Structure):
    # Mirrors SynthMovement in core/synth_core.h.
    _fields_ = [
        ("frameCount", ctypes.c_int),
        ("loopStart", ctypes.c_int),
        ("loopEnd", ctypes.c_int),
        ("amp", ctypes.POINTER(ctypes.c_float)),
        ("pitchCents", ctypes.POINTER(ctypes.c_float)),
        ("brightnessOct", ctypes.POINTER(ctypes.c_float)),
    ]


_SCHEMA = None


def param_info(identifier):
    """The schema entry (index, min, max, default, ...) for a parameter id.

    The schema is read once and cached. This is the only place Python reads
    parameter ranges, so nothing is hand-numbered.
    """
    global _SCHEMA
    if _SCHEMA is None:
        entries = json.loads(SCHEMA_PATH.read_text())["parameters"]
        _SCHEMA = {entry["id"]: entry for entry in entries}
    try:
        return _SCHEMA[identifier]
    except KeyError:
        raise KeyError(f"no parameter {identifier!r} in {SCHEMA_PATH.name}") from None


def param_index(identifier):
    """Index of a parameter by its stable id (its AU address)."""
    return int(param_info(identifier)["index"])


def _load_library(path=LIB_PATH):
    if not Path(path).exists():
        raise FileNotFoundError(
            f"{path} is missing. Build it with: clang++ -std=c++17 -O2 -dynamiclib -fPIC "
            "core/synth_core.cpp -o build/libsynthcore.dylib")
    lib = ctypes.CDLL(str(path))
    lib.synth_create.restype = ctypes.c_void_p
    lib.synth_create.argtypes = [ctypes.c_double]
    lib.synth_destroy.argtypes = [ctypes.c_void_p]
    lib.synth_set_sample_rate.argtypes = [ctypes.c_void_p, ctypes.c_double]
    lib.synth_reset.argtypes = [ctypes.c_void_p]
    lib.synth_get_param.restype = ctypes.c_float
    lib.synth_get_param.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.synth_set_param.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_float]
    lib.synth_set_timbre.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
    lib.synth_set_timbre_shaped.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
                                            ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
    lib.synth_set_timbre_moving.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
                                            ctypes.c_int, ctypes.POINTER(_SynthMovement)]
    lib.synth_set_timbre_full.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
                                          ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_int,
                                          ctypes.POINTER(ctypes.c_float), ctypes.c_int,
                                          ctypes.POINTER(_SynthMovement)]
    lib.synth_note_on.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    lib.synth_note_off.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.synth_render.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
    return lib


_LIB = None


def _lib():
    global _LIB
    if _LIB is None:
        _LIB = _load_library()
    return _LIB


def _float_ptr(array):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


class Core:
    """One synth instance. Use as a context manager, or call close()."""

    def __init__(self, sample_rate=44100.0):
        self.lib = _lib()
        self.sample_rate = float(sample_rate)
        self.handle = self.lib.synth_create(self.sample_rate)

    def close(self):
        if self.handle:
            self.lib.synth_destroy(self.handle)
            self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def set_param(self, identifier, value):
        self.lib.synth_set_param(self.handle, param_index(identifier), float(value))

    def get_param(self, identifier):
        return float(self.lib.synth_get_param(self.handle, param_index(identifier)))

    def reset(self):
        """Back to the just-created state for rendering (keeps parameters and timbre)."""
        self.lib.synth_reset(self.handle)

    def set_timbre(self, harmonics):
        h = np.ascontiguousarray(harmonics, dtype=np.float32)
        self.lib.synth_set_timbre(self.handle, _float_ptr(h), int(h.size))

    def set_timbre_shaped(self, harmonics, envelope):
        h = np.ascontiguousarray(harmonics, dtype=np.float32)
        e = np.ascontiguousarray(envelope, dtype=np.float32)
        self.lib.synth_set_timbre_shaped(self.handle, _float_ptr(h), int(h.size),
                                         _float_ptr(e), int(e.size))

    def set_timbre_full(self, harmonics, attack, movement):
        """Sustain harmonics, an optional attack spectrum (None for none), and movement."""
        h = np.ascontiguousarray(harmonics, dtype=np.float32)
        if attack is None:
            a_ptr, a_n = None, 0
            attack_keep = None
        else:
            attack_keep = np.ascontiguousarray(attack, dtype=np.float32)
            a_ptr, a_n = _float_ptr(attack_keep), int(attack_keep.size)
        track = None
        keep = []
        if movement is not None:
            amp = np.ascontiguousarray(movement.amp, dtype=np.float32)
            pitch = np.ascontiguousarray(movement.pitch_cents, dtype=np.float32)
            bright = np.ascontiguousarray(movement.brightness_oct, dtype=np.float32)
            keep = [amp, pitch, bright]
            track = _SynthMovement(frameCount=int(min(amp.size, pitch.size, bright.size)),
                                   loopStart=int(movement.loop_start),
                                   loopEnd=int(movement.loop_end),
                                   amp=_float_ptr(amp), pitchCents=_float_ptr(pitch),
                                   brightnessOct=_float_ptr(bright))
        self.lib.synth_set_timbre_full(self.handle, _float_ptr(h), int(h.size), a_ptr, a_n,
                                       None, 0, ctypes.byref(track) if track is not None else None)

    def set_timbre_moving(self, harmonics, movement):
        """`movement` is a neural.movement.Movement, or None for no movement."""
        h = np.ascontiguousarray(harmonics, dtype=np.float32)
        if movement is None:
            self.lib.synth_set_timbre_moving(self.handle, _float_ptr(h), int(h.size), None)
            return
        amp = np.ascontiguousarray(movement.amp, dtype=np.float32)
        pitch = np.ascontiguousarray(movement.pitch_cents, dtype=np.float32)
        bright = np.ascontiguousarray(movement.brightness_oct, dtype=np.float32)
        track = _SynthMovement(frameCount=int(min(amp.size, pitch.size, bright.size)),
                               loopStart=int(movement.loop_start),
                               loopEnd=int(movement.loop_end),
                               amp=_float_ptr(amp), pitchCents=_float_ptr(pitch),
                               brightnessOct=_float_ptr(bright))
        self.lib.synth_set_timbre_moving(self.handle, _float_ptr(h), int(h.size),
                                         ctypes.byref(track))

    def note_on(self, note, velocity=127):
        self.lib.synth_note_on(self.handle, int(note), int(velocity))

    def note_off(self, note):
        self.lib.synth_note_off(self.handle, int(note))

    def render(self, frames):
        out = np.zeros(int(frames), dtype=np.float32)
        self.lib.synth_render(self.handle, _float_ptr(out), int(frames))
        return out


def render_patch(core, patch, *, midi, velocity=127, hold_s, tail_s=0.0):
    """Renders one note on `core` with the patch applied.

    `patch` maps schema ids to values. The core is reset first, so the render
    depends only on the timbre, the patch and the note, never on earlier renders.
    The note is held for `hold_s`. If `tail_s` is above zero the key is released
    and the release tail is rendered too. Returns float32 mono samples.
    """
    core.reset()
    for identifier, value in patch.items():
        core.set_param(identifier, value)
    hold = int(round(hold_s * core.sample_rate))
    core.note_on(midi, velocity)
    if tail_s > 0.0:
        head = core.render(hold)
        core.note_off(midi)
        tail = core.render(int(round(tail_s * core.sample_rate)))
        return np.concatenate([head, tail])
    return core.render(hold)


def render_note(harmonics, *, movement=None, envelope=None, midi=69, velocity=127,
                seconds=8.0, sample_rate=44100.0, movement_amount=1.0):
    """A key held for `seconds`, with the neural source only (Source Mix 1).

    `movement` is a Movement or None. `envelope` is used only when there is no
    movement. `movement_amount` is the Movement parameter (0 to 1).
    """
    with Core(sample_rate) as core:
        if movement is not None:
            core.set_timbre_moving(harmonics, movement)
        elif envelope is not None:
            core.set_timbre_shaped(harmonics, envelope)
        else:
            core.set_timbre(harmonics)
        return render_patch(core, {"source_mix": 1.0, "movement": movement_amount},
                            midi=midi, velocity=velocity, hold_s=seconds)


def write_wav(path, samples, sample_rate):
    import wave
    x = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    pcm = (x * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sample_rate))
        w.writeframes(pcm.tobytes())
