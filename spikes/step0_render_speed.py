"""Step 0 spike: cost and pitch range of a DDSP-style harmonic decoder.

This is throwaway measurement code, not the generator. The decoder has random
weights, so it measures compute cost and whether the output pitch follows MIDI
note numbers. It says nothing about audio quality.

Decoder shape (harmonic + noise, as in DDSP):
  inputs  : MIDI pitch, velocity, latent z (16-dim, stands in for z_src)
  network : small MLP -> per-harmonic amplitudes and a noise-band gain
  synth   : additive harmonics at f0 = midi_to_hz(pitch), amplitudes
            mirrored at Nyquist-safe harmonic count, plus filtered noise

Run:  .venv/bin/python spikes/step0_render_speed.py
"""

import math
import time

import numpy as np
import torch
import torch.nn as nn

SR = 44100
SECONDS = 2.0
N_HARMONICS = 64
LATENT_DIM = 16
TEST_NOTES = list(range(21, 109))  # A0 .. C8, the piano range
REAL_TIME_BUDGET = 1.0  # seconds of compute per second of audio


def midi_to_hz(midi):
    return 440.0 * 2.0 ** ((midi - 69) / 12.0)


class HarmonicDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 + LATENT_DIM, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, N_HARMONICS + 1),
        )

    def forward(self, pitch, velocity, z, n_samples):
        # pitch, velocity: (B,) in MIDI and 0-1 units. z: (B, LATENT_DIM).
        p = (pitch - 60.0) / 48.0
        x = torch.cat([p[:, None], velocity[:, None], z], dim=1)
        out = self.net(x)
        harm_amp = torch.softmax(out[:, :N_HARMONICS], dim=1) * velocity[:, None]
        noise_gain = torch.sigmoid(out[:, N_HARMONICS]) * 0.05

        f0 = midi_to_hz(pitch)  # (B,)
        t = torch.arange(n_samples, device=pitch.device, dtype=torch.float32) / SR
        k = torch.arange(1, N_HARMONICS + 1, device=pitch.device, dtype=torch.float32)

        # Harmonics above Nyquist are dropped, so high notes stay alias-free.
        freqs = f0[:, None] * k[None, :]  # (B, H)
        mask = (freqs < SR / 2).float()
        amp = harm_amp * mask

        # Additive synthesis: sum over harmonics without a (B, H, T) tensor
        # for the full batch at once, chunked over harmonics.
        phase = 2 * math.pi * freqs[:, :, None] * t[None, None, :]
        audio = (amp[:, :, None] * torch.sin(phase)).sum(dim=1)
        noise = torch.randn(audio.shape, device=audio.device) * noise_gain[:, None]
        return audio + noise


def time_render(model, device, batch, n_samples, repeats=3):
    pitch = torch.tensor(np.random.uniform(40, 90, batch), dtype=torch.float32, device=device)
    vel = torch.full((batch,), 0.8, device=device)
    z = torch.randn(batch, LATENT_DIM, device=device)
    with torch.no_grad():
        model(pitch, vel, z, n_samples)  # warm-up
        if device.type == "mps":
            torch.mps.synchronize()
        best = float("inf")
        for _ in range(repeats):
            start = time.perf_counter()
            model(pitch, vel, z, n_samples)
            if device.type == "mps":
                torch.mps.synchronize()
            best = min(best, time.perf_counter() - start)
    return best


def pitch_check(model, device):
    """Renders each MIDI note alone and checks the spectral peak lands on f0."""
    errors = []
    with torch.no_grad():
        for midi in TEST_NOTES:
            pitch = torch.tensor([float(midi)], device=device)
            vel = torch.tensor([0.8], device=device)
            z = torch.zeros(1, LATENT_DIM, device=device)
            audio = model(pitch, vel, z, int(SR * 0.5)).cpu().numpy()[0]
            spectrum = np.abs(np.fft.rfft(audio * np.hanning(len(audio))))
            freqs = np.fft.rfftfreq(len(audio), 1.0 / SR)
            # Search the whole band below Nyquist. The strongest bin should sit
            # on some harmonic k*f0 (k >= 1). The fundamental itself may be
            # quieter than partials in an untrained model, so do not require k=1.
            peak_hz = freqs[np.argmax(spectrum)]
            f0 = midi_to_hz(midi)
            nearest = max(1, round(peak_hz / f0))
            err_cents = 1200 * math.log2(peak_hz / (nearest * f0))
            errors.append((midi, f0, peak_hz, abs(err_cents)))
    return errors


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    model = HarmonicDecoder().eval()

    devices = [torch.device("cpu")]
    if torch.backends.mps.is_available():
        devices.append(torch.device("mps"))

    print(f"Audio: {SECONDS:.1f} s at {SR} Hz, {N_HARMONICS} harmonics, budget {REAL_TIME_BUDGET}x real time")
    print()
    print(f"{'device':<6} {'batch':>5} {'render s':>9} {'rt factor':>10}  verdict")
    for device in devices:
        m = model.to(device)
        for batch in (1, 8, 32):
            seconds = time_render(m, device, batch, int(SR * SECONDS))
            # Real-time factor is per audio second for one voice.
            # We divide by the batch so it reads as cost per note.
            per_voice = seconds / batch
            rt = per_voice / SECONDS
            verdict = "ok" if rt < REAL_TIME_BUDGET else "over budget"
            print(f"{device.type:<6} {batch:>5} {seconds:>9.4f} {rt:>10.4f}  {verdict}")
        print()

    print("Pitch check on CPU (strongest spectral peak vs. nearest harmonic of f0)")
    model = model.to("cpu")
    errors = pitch_check(model, torch.device("cpu"))
    worst = max(errors, key=lambda e: e[3])
    within_50 = sum(1 for e in errors if e[3] <= 50)
    print(f"  notes tested     : {len(errors)} (MIDI {TEST_NOTES[0]}-{TEST_NOTES[-1]})")
    print(f"  within 50 cents  : {within_50}/{len(errors)}")
    print(f"  worst note       : MIDI {worst[0]} (f0 {worst[1]:.1f} Hz, peak {worst[2]:.1f} Hz, {worst[3]:.1f} cents)")


if __name__ == "__main__":
    main()
