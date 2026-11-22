"""Patch fit (step 2d): staged CMA-ES over the synth controls, against one note.

The neural source and its movement are fixed inputs. The fit adjusts synth
controls through the real C++ core, judged by a similarity loss against the
source note. It stops at a time cap, so every fit is bounded.

Two baselines are reported on the same note:
  - neutral: the controls the fit starts from (amplitude envelope neutral, so
    it does not fight the movement track). This is the fair baseline.
  - defaults: the schema's default controls, as the plugin plays them.
"""

from dataclasses import dataclass, field

import numpy as np

from .coreplay import Core, param_info, render_patch
from .model import SR, mr_stft_loss

# Controls the fit can free, in unit space. Groups are ablated together.
CORE_CONTROLS = ["env_attack", "env_decay", "env_sustain", "env_release",
                 "filter_cutoff", "filter_resonance", "drive",
                 "filter_env_amount", "lfo_pitch_depth", "lfo_rate"]
FILTER_ENV_CONTROLS = ["filter_env_attack", "filter_env_decay",
                       "filter_env_sustain", "filter_env_release"]
EXTRA_CONTROLS = FILTER_ENV_CONTROLS + ["lfo_filter_depth", "oscA_detune",
                                        "reverb_mix", "reverb_size"]

# Which controls the default fit frees. The ablation decides what is in here.
FIT_PARAMS = list(CORE_CONTROLS)

# Tools added in step 2d. Off by default, so they are only fitted when a
# configuration asks for them (see spikes/step2d_ablate.py).
FILTER_MODE_CONTROLS = ["filter_mode"]
SPECTRAL_CONTROLS = ["spectral_amount", "spectral_time"]

# Starting point for the fit: neutral amplitude envelope (fast attack, no decay,
# full sustain), so the movement track carries the loudness shape.
NEUTRAL = {"env_attack": 0.0, "env_decay": 0.0, "env_sustain": 1.0}

# Staged CMA-ES: groups of controls, small groups first, then everything.
# Sigma is in unit space.
STAGE_GROUPS = [
    (["env_attack", "env_decay", "env_sustain", "env_release", "filter_cutoff"], 0.4),
    (["filter_env_amount", "lfo_pitch_depth", "lfo_rate", "lfo_filter_depth"]
     + FILTER_ENV_CONTROLS, 0.4),
    (["filter_resonance", "drive", "reverb_mix", "reverb_size"], 0.3),
    (["oscA_detune"], 0.2),
    (FILTER_MODE_CONTROLS, 0.3),
    (SPECTRAL_CONTROLS, 0.3),
]

TAIL_S = 1.0          # release tail rendered after note-off (NSynth convention)
ENVELOPE_WEIGHT = 0.5


def envelope_db(x, rate=50):
    hop = SR // rate
    n = len(x) // hop
    rms = np.sqrt(np.mean(x[: n * hop].reshape(n, hop) ** 2, axis=1) + 1e-12)
    return np.maximum(20 * np.log10(rms), -60.0)


def similarity_loss(render, target):
    """Multi-resolution STFT on RMS-normalized audio, plus the envelope in dB.

    Lower is closer. The fit and the evaluation both use this, so it is the
    single definition of the match loss.
    """
    import torch

    n = min(len(render), len(target))
    x = render[:n] / (np.sqrt(np.mean(render[:n] ** 2)) + 1e-9)
    y = target[:n] / (np.sqrt(np.mean(target[:n] ** 2)) + 1e-9)
    spectral = float(mr_stft_loss(torch.as_tensor(x, dtype=torch.float32)[None],
                                  torch.as_tensor(y, dtype=torch.float32)[None]))
    env = float(np.mean(np.abs(envelope_db(render[:n]) - envelope_db(target[:n]))) / 20.0)
    return spectral + ENVELOPE_WEIGHT * env


def _unit(name, value):
    info = param_info(name)
    return (value - info["min"]) / (info["max"] - info["min"])


def unit_to_values(unit, controls=None):
    """Unit vector over `controls` -> schema values."""
    controls = FIT_PARAMS if controls is None else controls
    out = {}
    for name, u in zip(controls, unit):
        info = param_info(name)
        out[name] = info["min"] + float(np.clip(u, 0.0, 1.0)) * (info["max"] - info["min"])
    return out


def default_unit(controls=None):
    controls = FIT_PARAMS if controls is None else controls
    return np.array([_unit(n, param_info(n)["default"]) for n in controls])


def neutral_unit(controls=None):
    controls = FIT_PARAMS if controls is None else controls
    return np.array([_unit(n, NEUTRAL.get(n, param_info(n)["default"])) for n in controls])


def stages_for(controls):
    """The staged groups restricted to `controls`, then one stage with all of them.

    CMA-ES needs at least two dimensions in a stage, so a lone control is carried
    into the next group rather than optimized alone.
    """
    stages = []
    carry = []
    for names, sigma in STAGE_GROUPS:
        present = carry + [n for n in names if n in controls]
        carry = []
        if len(present) == 1:
            carry = present
        elif present:
            stages.append((present, sigma))
    stages.append((list(controls), 0.2))
    return stages


@dataclass
class FitResult:
    patch: dict            # schema id -> value, for every fitted control
    neutral_loss: float    # fair baseline: neutral controls, no fit
    defaults_loss: float   # schema default controls, no fit
    fitted_loss: float
    evals: int
    seconds: float
    moved: list = field(default_factory=list)  # ids that left their start point


def fit_patch(target, harmonics, movement, *, midi, velocity=110, seconds=30.0,
              controls=None, attack=None):
    """Fits the synth controls so the note matches `target` (16 kHz mono).

    `harmonics` and `movement` are the neural source, exactly as exported. The
    fit plays the note at `midi` and `velocity`, and stops after `seconds` of
    wall-clock time. Returns the best patch found, even if the cap was hit.
    """
    import time

    import cma

    controls = list(FIT_PARAMS if controls is None else controls)
    target = np.asarray(target, dtype=np.float32)
    hold_s = max(len(target) / SR - TAIL_S, 0.5 * len(target) / SR)
    tail_s = len(target) / SR - hold_s

    core = Core(SR)
    core.set_timbre_full(harmonics, attack, movement)
    core.set_param("source_mix", 1.0)

    def render(values):
        return render_patch(core, values, midi=midi, velocity=velocity,
                            hold_s=hold_s, tail_s=tail_s)

    start = neutral_unit(controls)
    neutral_loss = similarity_loss(render(unit_to_values(start, controls)), target)
    defaults_loss = similarity_loss(render(unit_to_values(default_unit(controls), controls)),
                                    target)
    best = {"x": start.copy(), "f": neutral_loss, "evals": 0}
    deadline = time.perf_counter() + seconds
    t0 = time.perf_counter()

    class _OutOfTime(Exception):
        pass

    def objective(unit):
        if time.perf_counter() > deadline:
            raise _OutOfTime
        unit = np.clip(unit, 0.0, 1.0)
        best["evals"] += 1
        value = similarity_loss(render(unit_to_values(unit, controls)), target)
        if value < best["f"]:
            best["f"], best["x"] = value, unit.copy()
        return value

    try:
        for stage, (names, sigma) in enumerate(stages_for(controls)):
            idx = [controls.index(n) for n in names]

            def sub(z, idx=idx):
                y = best["x"].copy()
                y[idx] = np.clip(z, 0.0, 1.0)
                return objective(y)

            es = cma.CMAEvolutionStrategy(best["x"][idx], sigma, {
                "bounds": [0.0, 1.0], "verbose": -9, "seed": stage,
                "maxfevals": 10 ** 6, "tolfun": 1e-12, "tolx": 1e-7})
            while not es.stop():
                candidates = es.ask()
                es.tell(candidates, [sub(z) for z in candidates])
    except _OutOfTime:
        pass

    values = unit_to_values(best["x"], controls)
    moved = [n for i, n in enumerate(controls) if abs(best["x"][i] - start[i]) > 0.05]
    return FitResult(
        patch={n: round(float(values[n]), 4) for n in controls},
        neutral_loss=float(neutral_loss),
        defaults_loss=float(defaults_loss),
        fitted_loss=float(best["f"]),
        evals=best["evals"],
        seconds=time.perf_counter() - t0,
        moved=moved,
    )
