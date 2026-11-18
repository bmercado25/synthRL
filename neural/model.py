"""Step 2b: conditioned neural decoder for the harmonic + noise source.

Encoder: log-mel spectrogram of a source note -> z (a pooled timbre vector).
Decoder: (pitch, velocity, z, time) -> per-frame harmonic distribution, loudness
         and noise level. These drive an additive synth at the target pitch.

The decoder is conditioned on pitch and velocity, so one model plays across
the keyboard. The export path averages its harmonic shape into the 64-number
timbre file that the plugin already loads, so the plugin needs no change.
"""

import math

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SR = 16000            # NSynth native rate
N_HARM = 64           # matches SYNTH_TIMBRE_MAX_HARMONICS
N_MELS = 64
N_FFT_MEL = 1024
HOP_MEL = 256
DEC_HOP = 128         # decoder frame hop, 8 ms at 16 kHz
Z_DIM = 32
NOTE_SECONDS = 4.0
NYQUIST_LIMIT = 0.45  # fraction of the sample rate, as in the core
EXPORT_MIDI = 60      # pitch used when exporting a pitch-independent timbre
EXPORT_VELOCITY = 127

_MEL_BASIS = None


def mel_basis():
    global _MEL_BASIS
    if _MEL_BASIS is None:
        _MEL_BASIS = torch.tensor(
            librosa.filters.mel(sr=SR, n_fft=N_FFT_MEL, n_mels=N_MELS), dtype=torch.float32)
    return _MEL_BASIS


def log_mel(wave):
    """(B, N) waveform -> (B, N_MELS, T) log-mel spectrogram."""
    window = torch.hann_window(N_FFT_MEL, device=wave.device)
    spec = torch.stft(wave, n_fft=N_FFT_MEL, hop_length=HOP_MEL, window=window,
                      return_complex=True).abs()
    mel = mel_basis().to(wave.device) @ spec
    return torch.log(mel + 1e-5)


class Encoder(nn.Module):
    def __init__(self, z_dim=Z_DIM, width=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(N_MELS, width, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv1d(width, width, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv1d(width, width, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv1d(width, width, 5, stride=2, padding=2), nn.GELU(),
        )
        self.proj = nn.Linear(width, z_dim)
        # Keeps z at unit scale. Without it z reached std ~180 and saturated
        # the decoder, so velocity had no effect (see build/neural/eval_run1.log).
        self.norm = nn.LayerNorm(z_dim)

    def forward(self, mel):
        mel = (mel + 5.0) / 4.0  # roughly centres log-mel values (about -11 to +2)
        h = self.conv(mel).mean(dim=-1)  # pool over time: one timbre vector
        return self.norm(self.proj(h))


class Decoder(nn.Module):
    def __init__(self, z_dim=Z_DIM, hidden=256):
        super().__init__()
        self.inp = nn.Linear(3 + z_dim, hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.harm = nn.Linear(hidden, N_HARM)
        self.loud = nn.Linear(hidden, 1)
        self.noise = nn.Linear(hidden, 1)

    def forward(self, midi, vel, z, frames):
        """Returns (dist (B,T,H), loud (B,T), noise (B,T)) for `frames` frames."""
        B = midi.shape[0]
        pitch = ((midi - 60.0) / 48.0)[:, None, None].expand(B, frames, 1)
        velo = (vel / 127.0)[:, None, None].expand(B, frames, 1)
        t = (torch.arange(frames, device=z.device, dtype=torch.float32) / frames)
        t = t[None, :, None].expand(B, frames, 1)
        zz = z[:, None, :].expand(B, frames, z.shape[1])
        h = self.inp(torch.cat([pitch, velo, t, zz], dim=-1))
        h, _ = self.gru(torch.tanh(h))
        dist = torch.softmax(self.harm(h), dim=-1)          # harmonic shape, sums to 1
        loud = F.softplus(self.loud(h)).squeeze(-1)         # total harmonic amplitude
        noise = F.softplus(self.noise(h)).squeeze(-1) * 0.01
        return dist, loud, noise


def synthesize(dist, loud, noise, midi, n_samples, noise_seed=0):
    """Additive harmonics at the note's pitch, plus filtered-by-envelope noise."""
    B, T, H = dist.shape
    f0 = 440.0 * torch.pow(2.0, (midi - 69.0) / 12.0)       # (B,)
    amp = dist * loud[..., None]                             # (B,T,H)
    amp = F.interpolate(amp.transpose(1, 2), size=n_samples, mode="linear",
                        align_corners=False)                 # (B,H,N)
    k = torch.arange(1, H + 1, device=dist.device, dtype=torch.float32)
    freqs = f0[:, None] * k[None, :]                         # (B,H)
    mask = (freqs < NYQUIST_LIMIT * SR).float()              # no harmonic above Nyquist
    amp = amp * mask[:, :, None]
    t = torch.arange(n_samples, device=dist.device, dtype=torch.float32) / SR
    phase = 2.0 * math.pi * freqs[:, :, None] * t[None, None, :]
    harmonic = (amp * torch.sin(phase)).sum(dim=1)           # (B,N)

    gen = torch.Generator().manual_seed(noise_seed)          # fixed seed: repeatable output
    white = torch.randn(B, n_samples, generator=gen).to(dist.device)
    noise_env = F.interpolate(noise[:, None, :], size=n_samples, mode="linear",
                              align_corners=False)[:, 0]
    return harmonic + white * noise_env


def mr_stft_loss(x, y, sizes=(512, 1024, 2048)):
    """Multi-resolution STFT loss: spectral convergence + log-magnitude L1."""
    total = 0.0
    for n in sizes:
        window = torch.hann_window(n, device=x.device)
        hop = n // 4
        X = torch.stft(x, n_fft=n, hop_length=hop, window=window, return_complex=True).abs()
        Y = torch.stft(y, n_fft=n, hop_length=hop, window=window, return_complex=True).abs()
        sc = torch.linalg.norm(Y - X, dim=(1, 2)) / (torch.linalg.norm(Y, dim=(1, 2)) + 1e-7)
        mag = (torch.log(X + 1e-7) - torch.log(Y + 1e-7)).abs().mean(dim=(1, 2))
        total = total + (sc + mag).mean()
    return total / len(sizes)


class NeuralSource(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def encode(self, wave):
        """(B, N) source waveforms at SR -> (B, Z_DIM) timbre vectors."""
        return self.encoder(log_mel(wave))

    def render(self, z, midi, vel, n_samples=int(SR * NOTE_SECONDS), noise_seed=0):
        frames = n_samples // DEC_HOP
        dist, loud, noise = self.decoder(midi, vel, z, frames)
        return synthesize(dist, loud, noise, midi, n_samples, noise_seed)


ENV_POINTS = 64  # matches SYNTH_TIMBRE_MAX_ENV_POINTS in core/synth_core.h


def export_from_model(model, wave_16k):
    """Pitch-independent timbre for the plugin, from one source note.

    Returns (harmonics, envelope):
      harmonics  64 harmonic amplitudes (peak 1): the loudness-weighted average
                 of the decoder's per-frame harmonic shape.
      envelope   ENV_POINTS loudness levels (peak 1) spread over the note. The
                 plugin applies it to the neural source, so the note's attack,
                 decay and sustain follow the decoder. Without it the export has
                 no loudness shape. On the 2b checkpoint, a constant level raised
                 the STFT loss from about 3.6 to about 13, so this matters most.
    """
    model.eval()
    with torch.no_grad():
        x = torch.as_tensor(wave_16k, dtype=torch.float32)[None]
        z = model.encode(x)
        midi = torch.tensor([float(EXPORT_MIDI)])
        vel = torch.tensor([float(EXPORT_VELOCITY)])
        frames = int(SR * NOTE_SECONDS) // DEC_HOP
        dist, loud, _ = model.decoder(midi, vel, z, frames)
        w = loud[0] / (loud[0].sum() + 1e-9)                 # loudness-weighted average
        profile = (w[:, None] * dist[0]).sum(dim=0).numpy()
        level = loud[0].numpy()
    profile = profile / profile.max()
    envelope = np.interp(np.linspace(0, frames - 1, ENV_POINTS), np.arange(frames), level)
    envelope = envelope / envelope.max()
    return profile.astype(np.float32), envelope.astype(np.float32)


def fit_to_note(wave_16k):
    """Any mono clip at SR -> the 4 s array the encoder was trained on.

    Clips longer than NOTE_SECONDS keep their loudest window (chosen by RMS, so
    a quiet tail does not win). Shorter clips are zero-padded. No pitch is needed.
    Raises ValueError for empty or silent input.
    """
    n = int(SR * NOTE_SECONDS)
    x = np.asarray(wave_16k, dtype=np.float32)
    if x.ndim != 1:
        raise ValueError("expected a mono signal")
    if x.size == 0 or float(np.abs(x).max()) < 1e-4:
        raise ValueError("signal is silent")
    if x.size <= n:
        return np.pad(x, (0, n - x.size))
    hop = SR // 4
    starts = list(range(0, x.size - n + 1, hop))
    if starts[-1] != x.size - n:
        starts.append(x.size - n)  # always consider the final window too
    energy = [float(np.mean(x[s:s + n] ** 2)) for s in starts]
    best = starts[int(np.argmax(energy))]
    return x[best:best + n]


def timbre_from_model(model, wave_16k):
    """Pitch-independent 64-harmonic timbre (peak 1), without the envelope."""
    return export_from_model(model, wave_16k)[0]
