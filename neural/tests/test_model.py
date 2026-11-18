"""Step 2b checks: the decoder, the loss, the timbre export and the training loop.

Run from the repo root:  .venv/bin/python -m pytest neural/tests
"""

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from neural import data
from neural.evaluate import pitch_error_cents
from neural.model import (DEC_HOP, N_HARM, NeuralSource, SR, mr_stft_loss,
                          synthesize, timbre_from_model)
from neural.timbre import read_timbre, write_timbre

N = SR * 4


def sine(freq, seconds=4.0):
    t = np.arange(int(SR * seconds)) / SR
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_loss_is_zero_for_identical_audio_and_positive_otherwise():
    x = torch.as_tensor(sine(220.0))[None]
    assert float(mr_stft_loss(x, x)) == pytest.approx(0.0, abs=1e-6)
    y = torch.as_tensor(sine(330.0))[None]
    assert float(mr_stft_loss(x, y)) > 0.1


def test_no_harmonic_above_nyquist_limit():
    # MIDI 120 is about 8.4 kHz, above the 7.2 kHz limit at 16 kHz, so every
    # harmonic is masked and, with no noise, the output is exactly silent.
    dist = torch.full((1, 10, N_HARM), 1.0 / N_HARM)
    loud = torch.ones(1, 10)
    zero_noise = torch.zeros(1, 10)
    out = synthesize(dist, loud, zero_noise, torch.tensor([120.0]), N)
    assert float(out.abs().max()) == 0.0


def test_render_shape_and_seeded_noise():
    torch.manual_seed(0)
    model = NeuralSource().eval()
    z = torch.zeros(2, 32)
    midi = torch.tensor([60.0, 72.0])
    vel = torch.tensor([100.0, 100.0])
    a = model.render(z, midi, vel, n_samples=N, noise_seed=3)
    b = model.render(z, midi, vel, n_samples=N, noise_seed=3)
    c = model.render(z, midi, vel, n_samples=N, noise_seed=4)
    assert a.shape == (2, N)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)
    assert torch.isfinite(a).all()


def test_timbre_export_matches_plugin_format(tmp_path):
    torch.manual_seed(1)
    model = NeuralSource().eval()
    wave = sine(261.6)
    profile = timbre_from_model(model, wave)
    assert profile.shape == (N_HARM,)
    assert profile.dtype == np.float32
    assert float(profile.max()) == pytest.approx(1.0)
    assert (profile >= 0).all()
    path = tmp_path / "model_timbre.json"
    write_timbre(path, profile, f0_hz=261.6, source_name="sine")
    assert np.allclose(read_timbre(path), profile, atol=1e-6)


def test_pitch_check_reads_a_sine_correctly():
    # A4 sine at 16 kHz, well inside the range the check is used for.
    cents = pitch_error_cents(sine(440.0), 69)
    assert abs(cents) < 5


def test_training_smoke_run(tmp_path):
    if not (data.ROOT / "valid" / "nsynth-valid" / "examples.json").exists():
        pytest.skip("NSynth valid split not downloaded (see CLAUDE.md)")
    from neural.train import train
    out = tmp_path / "smoke.pt"
    train(steps=2, batch=2, lr=1e-3, out=out, seed=0, log_every=1, eval_every=2)
    assert out.exists()
