"""Step 2c: measure how a source moves over time, straight from the audio.

Three tracks on a 50 Hz grid (one frame every 20 ms), up to 10 s:
  amp            smoothed RMS loudness, linear, peak 1
  pitch_cents    f0 contour in cents from the clip's median pitch, clipped to +-600;
                 all zeros when the clip has no clear pitch
  brightness_oct spectral centroid in octaves below the clip's brightest part
                 (90th percentile), 0 or below, floored at -4

Plus a loop region [loop_start, loop_end) in frames. The core plays from frame 0
once, then loops the region while a key is held. No model is involved, so this
works for unpitched sounds too.
"""

from dataclasses import dataclass

import librosa
import numpy as np

from .analyze import FMIN, FMAX, MIN_PITCHED_FRACTION, PITCH_CONFIDENCE
from .model import SR, loudest_window

RATE_HZ = 50.0
HOP = SR // 50                 # 320 samples = 20 ms at 16 kHz
MAX_SECONDS = 10.0
MAX_FRAMES = int(MAX_SECONDS * RATE_HZ)  # 500, matches SYNTH_MOVEMENT_MAX_FRAMES
MIN_LOOP_FRAMES = 50           # 1 s: shorter loops are not worth looping
BLEND_FRAMES = 5               # 100 ms crossfade at the loop's wrap point
PITCH_FRAME = 1024             # pYIN window: 2048 halves a 5 Hz vibrato's depth; 1024 keeps ~87%
PITCH_CLIP_CENTS = 600.0
BRIGHT_FLOOR_OCT = -4.0
LOUD_FRACTION = 0.1            # frames below this share of the peak count as silent
CENTROID_FLOOR_HZ = 20.0


@dataclass
class Movement:
    rate_hz: float
    loop_start: int            # frame index, inclusive
    loop_end: int              # frame index, exclusive
    amp: np.ndarray
    pitch_cents: np.ndarray
    brightness_oct: np.ndarray

    def __len__(self):
        return int(self.amp.shape[0])


def _on_grid(values, n):
    """Trims or edge-pads a per-frame track to exactly n frames."""
    values = np.asarray(values, dtype=np.float64)
    if values.shape[0] >= n:
        return values[:n]
    return np.pad(values, (0, n - values.shape[0]), mode="edge")


def _smooth3(values):
    """Three-frame moving average, edge-padded so the ends keep their level."""
    padded = np.pad(values, 1, mode="edge")
    return (padded[:-2] + padded[1:-1] + padded[2:]) / 3.0


def _fill_from(values, valid):
    """Replaces frames outside `valid` by linear interpolation from valid frames."""
    idx = np.arange(values.shape[0])
    return np.interp(idx, idx[valid], values[valid])


def _pitch_track(x, n):
    f0, _, prob = librosa.pyin(x, fmin=FMIN, fmax=FMAX, sr=SR,
                               frame_length=PITCH_FRAME, hop_length=HOP)
    f0 = _on_grid(np.nan_to_num(f0, nan=0.0), n)
    prob = _on_grid(prob, n)
    pitched = (f0 > 0.0) & (prob >= PITCH_CONFIDENCE)
    if pitched.mean() < MIN_PITCHED_FRACTION:
        return np.zeros(n)
    median = float(np.median(f0[pitched]))
    cents = 1200.0 * np.log2(np.maximum(f0, 1e-3) / median)
    cents = np.clip(cents, -PITCH_CLIP_CENTS, PITCH_CLIP_CENTS)
    return _fill_from(cents, pitched)


def _choose_loop(amp, pitch, bright):
    """Loop start and end (frames) whose seam has the smallest jump across tracks.

    The loop starts at or after the loudness peak, so the attack plays once. Each
    track is scaled by its spread so no single one dominates the jump.
    """
    n = amp.shape[0]
    attack = int(np.argmax(amp))
    if n < MIN_LOOP_FRAMES or attack > n - MIN_LOOP_FRAMES:
        return 0, n
    z = np.stack([amp, pitch, bright], axis=1)
    spread = z.std(axis=0)
    z = z / np.where(spread > 1e-6, spread, 1.0)
    best_jump, best_start, best_end = np.inf, attack, n
    for start in range(attack, n - MIN_LOOP_FRAMES + 1):
        ends = np.arange(start + MIN_LOOP_FRAMES, n + 1)
        jump = np.abs(z[ends - 1] - z[start]).sum(axis=1)
        k = int(np.argmin(jump))
        if jump[k] < best_jump - 1e-12:
            best_jump, best_start, best_end = float(jump[k]), start, int(ends[k])
    return best_start, best_end


def _blend_wrap(tracks, start, end):
    """Crossfades the last frames of the loop toward the frames just before its start.

    After the blend, the frame that wraps back to `start` continues the same
    material. Clips whose loop starts at frame 0 have nothing before it to blend
    with, so they wrap without a crossfade.
    """
    blend = min(BLEND_FRAMES, start)
    for track in tracks:
        for j in range(blend):
            w = (j + 1) / blend
            t, h = end - blend + j, start - blend + j
            track[t] = (1.0 - w) * track[t] + w * track[h]


def measure_movement(wave_16k):
    """Movement tracks for a mono clip at SR. Longer clips keep their loudest 10 s.

    Raises ValueError for empty or silent input.
    """
    x = np.asarray(wave_16k, dtype=np.float32)
    if x.ndim != 1:
        raise ValueError("expected a mono signal")
    if x.size == 0 or float(np.abs(x).max()) < 1e-4:
        raise ValueError("signal is silent")
    max_samples = int(MAX_SECONDS * SR)
    if x.size > max_samples:
        x = loudest_window(x, max_samples, hop=SR // 4)

    rms = librosa.feature.rms(y=x, frame_length=2 * HOP, hop_length=HOP, center=True)[0]
    n = min(rms.shape[0], MAX_FRAMES)
    amp = _smooth3(_on_grid(rms, n))
    amp = amp / max(float(amp.max()), 1e-9)

    pitch = _pitch_track(x, n)

    centroid = librosa.feature.spectral_centroid(y=x, sr=SR, n_fft=1024, hop_length=HOP)[0]
    centroid = np.maximum(_on_grid(centroid, n), CENTROID_FLOOR_HZ)
    loud = amp >= LOUD_FRACTION
    reference = float(np.percentile(centroid[loud], 90)) if loud.any() else float(centroid.max())
    bright = np.clip(np.log2(centroid / reference), BRIGHT_FLOOR_OCT, 0.0)
    if loud.any():
        bright = _fill_from(bright, loud)  # silent frames take the nearest audible value
    bright = _smooth3(bright)

    start, end = _choose_loop(amp, pitch, bright)
    tracks = [amp.copy(), pitch.copy(), bright.copy()]
    _blend_wrap(tracks, start, end)
    return Movement(rate_hz=RATE_HZ, loop_start=start, loop_end=end,
                    amp=tracks[0], pitch_cents=tracks[1], brightness_oct=tracks[2])
