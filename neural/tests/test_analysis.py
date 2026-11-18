"""Step 2a checks: analysis recovers a known timbre, and the file format round-trips.

Run from the repo root:  .venv/bin/python -m pytest neural/tests
"""

import math

import numpy as np
import pytest

from neural.analyze import HARMONICS, analyze
from neural.render import render_note
from neural.timbre import read_timbre, write_timbre

SR = 44100


def harmonic_signal(f0, amps, seconds=2.0):
    t = np.arange(int(SR * seconds)) / SR
    y = np.zeros_like(t)
    for k, a in enumerate(amps, start=1):
        y += a * np.sin(2 * np.pi * k * f0 * t)
    return (y / np.max(np.abs(y)) * 0.8).astype(np.float32)


def cents(a, b):
    return 1200 * math.log2(a / b)


def test_sine_source_gives_fundamental_only():
    y = harmonic_signal(220.0, [1.0])
    result = analyze(y, SR)
    assert abs(cents(result.f0_hz, 220.0)) < 5
    assert result.harmonics[0] == pytest.approx(1.0)
    assert np.max(result.harmonics[1:]) < 0.02


def test_known_harmonic_series_is_recovered():
    amps = [1.0 / k ** 1.5 for k in range(1, 13)]  # 12 harmonics, falling off
    y = harmonic_signal(196.0, amps)
    result = analyze(y, SR)
    assert abs(cents(result.f0_hz, 196.0)) < 5
    target = np.zeros(HARMONICS)
    target[:12] = np.array(amps) / amps[0]
    err = np.abs(result.harmonics[:12] - target[:12])
    assert err.max() < 0.05, f"largest harmonic error {err.max():.3f}"


def test_round_trip_keeps_the_timbre():
    # Analyse a source, render a note with the reference renderer at the
    # source pitch, analyse that note again: the timbre should come back.
    amps = [1.0, 0.5, 0.25, 0.125]
    source = harmonic_signal(261.6, amps)
    first = analyze(source, SR)
    rendered = render_note(first.harmonics, midi=60, velocity=127, seconds=2.0, sample_rate=SR)
    second = analyze(rendered.astype(np.float32), SR)
    assert np.max(np.abs(first.harmonics[:4] - second.harmonics[:4])) < 0.05


def test_no_pitch_is_rejected():
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(SR).astype(np.float32) * 0.3
    with pytest.raises(ValueError):
        analyze(noise, SR)


def test_silence_is_rejected():
    with pytest.raises(ValueError):
        analyze(np.zeros(SR, dtype=np.float32), SR)


def test_timbre_file_round_trip(tmp_path):
    harmonics = np.zeros(HARMONICS, dtype=np.float32)
    harmonics[:3] = [1.0, 0.4, 0.1]
    path = tmp_path / "t.json"
    write_timbre(path, harmonics, f0_hz=261.6, source_name="test.wav")
    back = read_timbre(path)
    assert np.allclose(back, harmonics, atol=1e-6)


def test_timbre_file_rejects_wrong_format(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"format": "something-else", "version": 1, "harmonics": []}')
    with pytest.raises(ValueError):
        read_timbre(path)
