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


def _have_splits():
    needed = ["train", "valid"]
    return all((data.split_dir(s) / "examples.json").exists() for s in needed)


def test_training_smoke_run(tmp_path):
    if not _have_splits():
        pytest.skip("NSynth train and valid splits not downloaded (see CLAUDE.md)")
    from neural.train import train
    out = tmp_path / "smoke.pt"
    train(steps=2, batch=2, lr=1e-3, out=out, seed=0, log_every=1, eval_every=2,
          resume=False)
    assert out.exists()


def test_training_resumes_from_last_checkpoint(tmp_path):
    if not _have_splits():
        pytest.skip("NSynth train and valid splits not downloaded (see CLAUDE.md)")
    from neural.train import last_path, train
    out = tmp_path / "resume.pt"
    train(steps=2, batch=2, lr=1e-3, out=out, seed=0, log_every=1, eval_every=2,
          resume=False)
    first = torch.load(last_path(out), map_location="cpu")
    assert first["step"] == 2
    # Resuming to 4 steps must continue from step 2, not start over.
    train(steps=4, batch=2, lr=1e-3, out=out, seed=0, log_every=1, eval_every=2,
          resume=True)
    second = torch.load(last_path(out), map_location="cpu")
    assert second["step"] == 4


def test_timbre_envelope_round_trip_and_v1_still_reads(tmp_path):
    from neural.timbre import read_envelope
    profile = np.zeros(N_HARM, dtype=np.float32)
    profile[0] = 1.0
    env = np.linspace(0.0, 1.0, 64, dtype=np.float32)
    path = tmp_path / "v2.json"
    write_timbre(path, profile, f0_hz=261.6, source_name="sine", envelope=env)
    assert np.allclose(read_envelope(path), env, atol=1e-6)
    assert np.allclose(read_timbre(path), profile, atol=1e-6)

    # A version 1 file (no envelope) still reads, and reports no envelope.
    v1 = tmp_path / "v1.json"
    write_timbre(v1, profile, f0_hz=261.6, source_name="sine")
    assert read_envelope(v1) is None
    assert np.allclose(read_timbre(v1), profile, atol=1e-6)


def test_envelope_rejects_too_many_points(tmp_path):
    profile = np.zeros(N_HARM, dtype=np.float32)
    profile[0] = 1.0
    with pytest.raises(ValueError):
        write_timbre(tmp_path / "bad.json", profile, f0_hz=261.6, source_name="x",
                     envelope=np.ones(65, dtype=np.float32))


def test_any_sound_exports_without_a_pitch(tmp_path):
    from neural.model import ENV_POINTS, export_from_model, fit_to_note
    from neural.timbre import read_envelope
    torch.manual_seed(2)
    model = NeuralSource().eval()
    rng = np.random.default_rng(3)
    clips = {
        "noise": (0.3 * rng.standard_normal(SR * 2)).astype(np.float32),
        "click": np.zeros(SR // 10, dtype=np.float32),
        "long": (0.3 * rng.standard_normal(SR * 30)).astype(np.float32),
    }
    clips["click"][10] = 0.9
    for name, wave in clips.items():
        harmonics, envelope = export_from_model(model, fit_to_note(wave))
        assert harmonics.shape == (N_HARM,), name
        assert np.isfinite(harmonics).all(), name
        assert float(harmonics.max()) == pytest.approx(1.0), name
        assert envelope.shape == (ENV_POINTS,), name
        path = tmp_path / f"{name}.json"
        write_timbre(path, harmonics, f0_hz=None, source_name=name, envelope=envelope)
        assert np.allclose(read_timbre(path), harmonics, atol=1e-6), name
        assert np.allclose(read_envelope(path), envelope, atol=1e-6), name


def test_fit_to_note_pads_short_and_picks_loudest_window():
    from neural.model import NOTE_SECONDS, fit_to_note
    n = int(SR * NOTE_SECONDS)
    short = fit_to_note(np.full(SR, 0.5, dtype=np.float32))
    assert short.shape == (n,) and short[SR:].max() == 0.0
    # A loud burst at 20 s should win over the quiet start of a 30 s clip.
    long = np.full(SR * 30, 0.01, dtype=np.float32)
    long[SR * 20:SR * 20 + SR] = 0.8
    fitted = fit_to_note(long)
    assert fitted.shape == (n,) and fitted.max() == pytest.approx(0.8)


def test_silence_is_rejected():
    from neural.model import fit_to_note
    with pytest.raises(ValueError):
        fit_to_note(np.zeros(SR, dtype=np.float32))


def test_null_f0_round_trips(tmp_path):
    profile = np.zeros(N_HARM, dtype=np.float32)
    profile[0] = 1.0
    path = tmp_path / "unpitched.json"
    write_timbre(path, profile, f0_hz=None, source_name="noise")
    assert np.allclose(read_timbre(path), profile, atol=1e-6)
