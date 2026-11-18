"""Reference additive renderer for a timbre.

Used by tests to check analysis round-trips. It is not the plugin's renderer:
the core uses bandlimited wavetables and an envelope, and this does neither.
"""

import numpy as np

from .analyze import HARMONICS

MASTER_AMPLITUDE = 0.5  # matches kMasterAmplitude in core/synth_core.cpp


def midi_to_hz(midi):
    return 440.0 * 2.0 ** ((midi - 69) / 12.0)


def render_note(harmonics, midi, velocity, seconds, sample_rate=44100):
    """Sums the harmonics at the note's pitch, skipping any above 0.45 * sample rate."""
    harmonics = np.asarray(harmonics, dtype=np.float64)
    if harmonics.shape != (HARMONICS,):
        raise ValueError(f"expected {HARMONICS} harmonics")
    f0 = midi_to_hz(midi)
    n = int(round(seconds * sample_rate))
    t = np.arange(n) / sample_rate
    out = np.zeros(n)
    limit = 0.45 * sample_rate
    for k in range(1, HARMONICS + 1):
        freq = k * f0
        if freq >= limit:
            break
        out += harmonics[k - 1] * np.sin(2 * np.pi * freq * t)
    peak = np.max(np.abs(out))
    if peak > 0:
        out /= peak
    return (MASTER_AMPLITUDE * (velocity / 127.0)) * out


def render_export(harmonics, envelope, midi, velocity, seconds, sample_rate=44100):
    """render_note with a loudness envelope over the note (linear between points,
    held at the last point), as the core applies it to the neural source."""
    y = render_note(harmonics, midi, velocity, seconds, sample_rate)
    env = np.asarray(envelope, dtype=np.float64)
    t = np.linspace(0.0, 1.0, len(y))
    level = np.interp(t, np.linspace(0.0, 1.0, len(env)), env) if len(env) > 1 else env[0]
    return y * level
