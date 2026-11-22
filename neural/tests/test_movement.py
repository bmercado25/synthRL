"""Step 2c checks: movement measurement, the version 3 timbre file, the export command,
and the objective check through the real core.

Run from the repo root:  .venv/bin/python -m pytest neural/tests
"""

from pathlib import Path

import numpy as np
import pytest

from neural.coreplay import LIB_PATH, write_wav
from neural.export import main as export_main
from neural.model import SR
from neural.movement import MIN_LOOP_FRAMES, MAX_FRAMES, measure_movement
from neural.movement_eval import checks, dominant_hz, evaluate
from neural.timbre import read_envelope, read_movement, read_timbre, write_timbre


def tremolo(seconds=3.0, rate=6.0):
    t = np.arange(int(SR * seconds)) / SR
    tone = sum(np.sin(2 * np.pi * 220 * k * t) / k for k in range(1, 5))
    return ((1.0 + 0.8 * np.sin(2 * np.pi * rate * t)) * tone * 0.3).astype(np.float32)


def test_amplitude_track_follows_tremolo():
    m = measure_movement(tremolo())
    assert m.rate_hz == 50.0
    assert m.amp.max() == pytest.approx(1.0)
    assert dominant_hz(m.amp) == pytest.approx(6.0, rel=0.05)


def test_pitched_clip_has_small_pitch_track_and_brightness_is_bounded():
    m = measure_movement(tremolo())
    assert np.abs(m.pitch_cents).max() < 10.0           # a steady tone barely moves
    assert m.brightness_oct.max() <= 0.0
    assert m.brightness_oct.min() >= -4.0


def test_unpitched_noise_has_no_pitch_track():
    rng = np.random.default_rng(0)
    noise = (rng.standard_normal(SR * 2) * 0.1).astype(np.float32)
    m = measure_movement(noise)
    assert np.all(m.pitch_cents == 0.0)
    assert m.amp.max() == pytest.approx(1.0)


def test_loop_is_after_the_attack_and_at_least_one_second_long():
    t = np.arange(SR * 3) / SR
    decay = np.exp(-2.0 * t) * np.sin(2 * np.pi * 330 * t)
    m = measure_movement(decay.astype(np.float32))
    assert m.loop_start >= int(np.argmax(m.amp))
    assert m.loop_end - m.loop_start >= MIN_LOOP_FRAMES
    assert m.loop_end <= len(m) <= MAX_FRAMES


def test_clip_under_one_second_loops_whole():
    m = measure_movement(tremolo(seconds=0.5))
    assert (m.loop_start, m.loop_end) == (0, len(m))


def test_timbre_v3_round_trips_movement(tmp_path):
    harm = np.zeros(64, dtype=np.float32)
    harm[0] = 1.0
    m = measure_movement(tremolo())
    path = tmp_path / "v3.json"
    write_timbre(path, harm, f0_hz=220.0, source_name="t.wav", movement=m)
    back = read_movement(path)
    assert back.loop_start == m.loop_start and back.loop_end == m.loop_end
    np.testing.assert_allclose(back.amp, m.amp, atol=1e-5)
    np.testing.assert_allclose(back.brightness_oct, m.brightness_oct, atol=1e-5)
    np.testing.assert_allclose(read_timbre(path), harm)
    assert read_envelope(path) is None


def test_v2_files_still_load_without_movement(tmp_path):
    harm = np.zeros(64, dtype=np.float32)
    harm[0] = 1.0
    path = tmp_path / "v2.json"
    write_timbre(path, harm, f0_hz=220.0, source_name="t.wav", envelope=[0.2, 1.0])
    assert read_movement(path) is None
    np.testing.assert_allclose(read_envelope(path), [0.2, 1.0])


def test_envelope_and_movement_cannot_both_be_written(tmp_path):
    harm = np.zeros(64, dtype=np.float32)
    harm[0] = 1.0
    with pytest.raises(ValueError):
        write_timbre(tmp_path / "x.json", harm, f0_hz=220.0, source_name="x",
                     envelope=[1.0], movement=measure_movement(tremolo()))


def test_export_command_writes_movement_unless_disabled(tmp_path):
    wav = tmp_path / "clip.wav"
    write_wav(wav, tremolo(), SR)
    with_move = tmp_path / "with.json"
    without = tmp_path / "without.json"
    assert export_main([str(wav), str(with_move)]) == 0
    assert export_main([str(wav), str(without), "--no-movement"]) == 0
    assert read_movement(with_move) is not None
    assert read_movement(without) is None


@pytest.mark.skipif(not Path(LIB_PATH).exists(),
                    reason="build/libsynthcore.dylib missing (scripts/test.sh builds it)")
def test_core_plays_the_source_movement():
    rows = evaluate(checkpoint=None)
    failed = [label for label, _, ok in checks(rows) if not ok]
    assert not failed, failed
