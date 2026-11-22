"""Step 2d checks: the version 4 patch block, the fit, and the export command's --fit.

The fit tests need build/libsynthcore.dylib (scripts/test.sh builds it) and are
skipped without it. Budgets are kept to a few seconds.

Run from the repo root:  .venv/bin/python -m pytest neural/tests/test_fit.py
"""

import json

import numpy as np
import pytest

from neural.coreplay import LIB_PATH, param_info, write_wav
from neural.export import main as export_main
from neural.fit import FIT_PARAMS, fit_patch, similarity_loss
from neural.model import SR
from neural.timbre import read_patch, read_timbre, write_timbre

needs_core = pytest.mark.skipif(not LIB_PATH.exists(),
                                reason="build/libsynthcore.dylib missing (scripts/test.sh builds it)")


def harmonics():
    return np.array([1.0 / k ** 1.2 for k in range(1, 17)] + [0.0] * 48, dtype=np.float32)


def test_patch_round_trips_through_version_4(tmp_path):
    path = tmp_path / "t.json"
    patch = {"drive": 0.3, "env_attack": 0.2, "filter_cutoff": 0.9}
    write_timbre(path, harmonics(), f0_hz=220.0, source_name="x.wav", patch=patch)
    doc = json.loads(path.read_text())
    assert doc["version"] == 5
    assert doc["patch"]["drive"]["index"] == int(param_info("drive")["index"])
    got = read_patch(path)
    for key, value in patch.items():
        assert got[key] == pytest.approx(value, abs=1e-6)
    assert read_timbre(path).shape == (64,)


def test_attack_spectrum_round_trips_through_version_5(tmp_path):
    from neural.timbre import read_attack
    path = tmp_path / "t.json"
    attack = np.linspace(1.0, 0.1, 64).astype(np.float32)
    write_timbre(path, harmonics(), f0_hz=220.0, source_name="x.wav", attack_harmonics=attack)
    assert json.loads(path.read_text())["version"] == 5
    assert np.allclose(read_attack(path), attack, atol=1e-6)


def test_files_without_an_attack_spectrum_read_as_none(tmp_path):
    from neural.timbre import read_attack
    path = tmp_path / "t.json"
    write_timbre(path, harmonics(), f0_hz=220.0, source_name="x.wav")
    assert read_attack(path) is None


def test_files_without_a_patch_read_as_none(tmp_path):
    path = tmp_path / "t.json"
    write_timbre(path, harmonics(), f0_hz=220.0, source_name="x.wav")
    assert read_patch(path) is None


def test_patch_rejects_unknown_ids_and_out_of_range_values(tmp_path):
    path = tmp_path / "t.json"
    with pytest.raises(KeyError):
        write_timbre(path, harmonics(), f0_hz=220.0, source_name="x.wav",
                     patch={"not_a_parameter": 0.5})
    with pytest.raises(ValueError):
        write_timbre(path, harmonics(), f0_hz=220.0, source_name="x.wav",
                     patch={"drive": 2.0})


def test_read_rejects_a_patch_index_that_disagrees_with_the_schema(tmp_path):
    path = tmp_path / "t.json"
    write_timbre(path, harmonics(), f0_hz=220.0, source_name="x.wav", patch={"drive": 0.1})
    doc = json.loads(path.read_text())
    doc["patch"]["drive"]["index"] = 0
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError):
        read_patch(path)


def _tone(seconds=1.0, midi_hz=220.0):
    t = np.arange(int(SR * seconds)) / SR
    x = sum(np.sin(2 * np.pi * midi_hz * k * t) / k ** 1.2 for k in range(1, 9))
    env = np.exp(-3.0 * t)
    return (0.3 * x * env).astype(np.float32)


@needs_core
def test_fit_does_not_end_worse_than_the_defaults():
    target = _tone()
    result = fit_patch(target, harmonics(), None, midi=57, seconds=3.0)
    assert set(result.patch) == set(FIT_PARAMS)
    for name, value in result.patch.items():
        info = param_info(name)
        assert info["min"] <= value <= info["max"]
    assert result.fitted_loss <= result.neutral_loss + 1e-9
    assert result.evals > 0


@needs_core
def test_fit_stops_at_its_time_cap():
    result = fit_patch(_tone(), harmonics(), None, midi=57, seconds=1.0)
    assert result.seconds < 5.0


def test_similarity_loss_is_zero_for_identical_audio():
    x = _tone(0.5)
    assert similarity_loss(x, x) == pytest.approx(0.0, abs=1e-6)


@needs_core
def test_export_fit_writes_a_version_4_patch(tmp_path):
    from neural.fixtures import tremolo_clip
    wav = tmp_path / "clip.wav"
    write_wav(wav, tremolo_clip(seconds=2.0), SR)
    out = tmp_path / "timbre.json"
    assert export_main([str(wav), str(out), "--fit", "--fit-seconds", "2"]) == 0
    doc = json.loads(out.read_text())
    assert doc["version"] == 5
    assert set(doc["patch"]) == set(FIT_PARAMS)
    assert "movement" in doc  # the fit does not replace the movement
