"""Analysis-resynthesis source: turns a pitched sample into a harmonic timbre.

Step 2a. No training. The sample's fundamental is estimated with pYIN, the
spectrum is averaged over its sustained frames, and the magnitude at each
harmonic of f0 becomes one entry of the timbre. The core then plays that
timbre at any pitch.

Known limits (fixed in step 2b):
  - Only magnitudes are kept. The sample's phases are discarded.
  - Noise and transients are not modelled.
  - One timbre per sample, with no temporal change.
"""

from dataclasses import dataclass

import librosa
import numpy as np

HARMONICS = 64  # must match SYNTH_TIMBRE_MAX_HARMONICS in core/synth_core.h
N_FFT = 8192
HOP = 2048
FMIN = 40.0
FMAX = 4000.0  # was 2000: pYIN halved notes above 2 kHz (labelled NSynth vocal at MIDI 102)
ATTACK_WINDOW_S = 0.1  # the attack spectrum is measured over the first 100 ms
SUSTAIN_THRESHOLD = 0.25  # keep frames whose RMS is at least this fraction of the peak
PITCH_CONFIDENCE = 0.5  # pYIN probability a frame must reach to count as pitched
MIN_PITCHED_FRACTION = 0.5  # reject the sample if fewer frames than this are pitched


@dataclass
class SourceAnalysis:
    sample_rate: int
    f0_hz: float
    harmonics: np.ndarray  # shape (HARMONICS,), peak-normalized, fundamental first
    # Spectrum of the first ATTACK_WINDOW_S seconds, on the same scale as `harmonics`
    # (divided by the sustain's raw peak). None when the clip is too short to tell.
    attack_harmonics: np.ndarray | None = None


def analyze(samples: np.ndarray, sample_rate: int) -> SourceAnalysis:
    """Estimates f0 and the harmonic timbre of a mono signal.

    Raises ValueError if the signal has no clear pitch.
    """
    y = np.asarray(samples, dtype=np.float32)
    if y.ndim != 1:
        raise ValueError("expected a mono signal")

    f0, _, prob = librosa.pyin(y, fmin=FMIN, fmax=FMAX, sr=sample_rate,
                               frame_length=N_FFT, hop_length=HOP)
    # pYIN's voiced flag marks most noise frames as voiced, so gate on its
    # probability instead: noise scores about 0.01, a steady tone about 0.98.
    pitched = np.isfinite(f0) & (prob >= PITCH_CONFIDENCE)
    if pitched.mean() < MIN_PITCHED_FRACTION:
        raise ValueError("no clear pitch found")
    f0_hz = float(np.median(f0[pitched]))

    spectrum = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP))  # (bins, frames)
    freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=N_FFT)
    rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP)[0]
    frames = rms.shape[0]
    spectrum = spectrum[:, :frames]
    sustained = rms >= SUSTAIN_THRESHOLD * rms.max()
    if not sustained.any():
        raise ValueError("signal is silent")
    mean_mag = spectrum[:, sustained].mean(axis=1)

    nyquist_limit = 0.45 * sample_rate
    amps = np.zeros(HARMONICS, dtype=np.float64)
    half_window = 0.25 * f0_hz
    for k in range(1, HARMONICS + 1):
        centre = k * f0_hz
        if centre >= nyquist_limit:
            break
        window = (freqs >= centre - half_window) & (freqs <= centre + half_window)
        if window.any():
            amps[k - 1] = mean_mag[window].max()

    peak = amps.max()
    if peak <= 0.0:
        raise ValueError("no harmonic energy found")

    attack_harmonics = None
    early = np.arange(frames) * HOP / sample_rate < ATTACK_WINDOW_S
    if early.any():
        early_mag = spectrum[:, early].mean(axis=1)
        attack_amps = np.zeros(HARMONICS, dtype=np.float64)
        for k in range(1, HARMONICS + 1):
            centre = k * f0_hz
            if centre >= nyquist_limit:
                break
            window = (freqs >= centre - half_window) & (freqs <= centre + half_window)
            if window.any():
                attack_amps[k - 1] = early_mag[window].max()
        attack_harmonics = (attack_amps / peak).astype(np.float32)

    return SourceAnalysis(sample_rate=sample_rate, f0_hz=f0_hz,
                          harmonics=(amps / peak).astype(np.float32),
                          attack_harmonics=attack_harmonics)


def analyze_file(path: str) -> SourceAnalysis:
    samples, sr = librosa.load(path, sr=None, mono=True)
    return analyze(samples, sr)
