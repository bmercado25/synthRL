"""Step 2c objective check: does the synth carry a source's movement?

Synthetic references with a known character (tremolo, vibrato, brightness sweep,
rhythmic pulse) go through the step 2a analysis and measure_movement. The real
core then plays each one by holding a key for 8 s at movement 1, 0 and with the
v2 export (decoder envelope, no movement). Each render is measured with the same
code, and its tracks are compared with the reference tracks, expanded the way
the core loops them.

    .venv/bin/python -m neural.movement_eval [--checkpoint build/...pt]

Needs build/libsynthcore.dylib (scripts/test.sh builds it).
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from .analyze import analyze
from .coreplay import render_note
from .model import SR, fit_to_note
from .movement import RATE_HZ, measure_movement

HOLD_SECONDS = 8.0
REFERENCE_SECONDS = 3.0
TONE_HZ = 220.0
DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[1] / "data" / "checkpoints" / "nsynth_2b_full.pt"


def _tone(phase, harm):
    """Harmonics of a phase track (radians): harm[k] is the weight of harmonic k+1."""
    k = np.arange(1, len(harm) + 1)[:, None]
    return (harm[:, None] * np.sin(k * phase[None, :])).sum(axis=0)


def make_references():
    """Name -> (signal at SR, what it does). Peak is set to 0.8 for every reference."""
    n = int(SR * REFERENCE_SECONDS)
    t = np.arange(n) / SR
    base = 2 * np.pi * TONE_HZ * t
    harm = 1.0 / np.arange(1, 5)  # a fixed four-harmonic tone for amplitude and pulse cases

    refs = {}
    refs["tremolo"] = (1.0 + 0.8 * np.sin(2 * np.pi * 6.0 * t)) * _tone(base, harm)

    cents = 30.0 * np.sin(2 * np.pi * 5.0 * t)
    inst_phase = 2 * np.pi * np.cumsum(TONE_HZ * 2.0 ** (cents / 1200.0)) / SR
    refs["vibrato"] = _tone(inst_phase, harm)

    k = np.arange(1, 25)[:, None]
    slope = 2.0 + (0.3 - 2.0) * t / t[-1]  # harmonic falloff eases from dark to bright
    amps = k ** (-slope[None, :])
    refs["sweep"] = (amps * np.sin(k * base[None, :])).sum(axis=0)

    gate = (np.sin(2 * np.pi * 4.0 * t) > 0).astype(float)
    refs["pulse"] = (0.1 + 0.9 * gate) * _tone(base, harm)

    out = {}
    for name, x in refs.items():
        x = x / np.max(np.abs(x)) * 0.8
        out[name] = x.astype(np.float32)
    return out


def played_track(track, loop_start, loop_end, n_frames):
    """The frames the core plays for a track: once to loop_start, then the loop repeats."""
    k = np.arange(n_frames)
    wrapped = loop_start + (k - loop_start) % (loop_end - loop_start)
    index = np.where(k >= loop_end, wrapped, k)
    return track[index]


def dominant_hz(x, rate=RATE_HZ):
    """Frequency of the strongest modulation in a frame track (DC and below 0.5 Hz removed)."""
    x = np.asarray(x, dtype=np.float64)
    x = (x - x.mean()) * np.hanning(x.size)
    n = 1 << int(np.ceil(np.log2(x.size * 8)))
    spec = np.abs(np.fft.rfft(x, n))
    freqs = np.fft.rfftfreq(n, 1.0 / rate)
    spec[freqs < 0.5] = 0.0
    return float(freqs[int(np.argmax(spec))])


def correlation(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.std() < 1e-9 or b.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def evaluate(checkpoint=None):
    """Runs every case and returns a list of result rows (dicts)."""
    refs = make_references()
    v2_model = None
    if checkpoint is not None and Path(checkpoint).exists():
        from .evaluate import load_model
        v2_model, _ = load_model(checkpoint)

    rows = []
    for name, ref in refs.items():
        ref_track = measure_movement(ref)
        harmonics = analyze(ref, SR).harmonics
        loop_s, loop_e = ref_track.loop_start, ref_track.loop_end

        moved = render_note(harmonics, movement=ref_track, seconds=HOLD_SECONDS,
                            sample_rate=SR, movement_amount=1.0)
        static = render_note(harmonics, movement=ref_track, seconds=HOLD_SECONDS,
                             sample_rate=SR, movement_amount=0.0)
        m_moved = measure_movement(moved)
        m_static = measure_movement(static)
        n = len(m_moved.amp)

        expected_amp = played_track(ref_track.amp, loop_s, loop_e, n)
        expected_bright = played_track(ref_track.brightness_oct, loop_s, loop_e, n)
        expected_pitch = played_track(ref_track.pitch_cents, loop_s, loop_e, n)

        row = {"case": name,
               "amp_corr": correlation(m_moved.amp, expected_amp),
               "bright_corr": correlation(m_moved.brightness_oct, expected_bright),
               "static_amp_corr": correlation(m_static.amp, expected_amp)}
        if name in ("tremolo", "pulse"):
            row["amp_rate_hz"] = dominant_hz(m_moved.amp)
            row["amp_rate_ref_hz"] = 6.0 if name == "tremolo" else 4.0
        if name == "vibrato":
            row["vib_rate_hz"] = dominant_hz(m_moved.pitch_cents)
            ref_depth = np.std(ref_track.pitch_cents[loop_s:loop_e]) * np.sqrt(2)
            row["vib_depth_cents"] = float(np.std(m_moved.pitch_cents) * np.sqrt(2))
            row["vib_depth_ref_cents"] = float(ref_depth)
            row["pitch_corr"] = correlation(m_moved.pitch_cents, expected_pitch)

        if v2_model is not None:
            v2_harm, v2_env = _v2_export(v2_model, ref)
            v2 = render_note(v2_harm, envelope=v2_env, seconds=HOLD_SECONDS, sample_rate=SR)
            m_v2 = measure_movement(v2)
            row["v2_amp_corr"] = correlation(m_v2.amp, expected_amp)
            row["v2_bright_corr"] = correlation(m_v2.brightness_oct, expected_bright)
        rows.append(row)
    return rows


def _v2_export(model, ref):
    from .model import export_from_model
    return export_from_model(model, fit_to_note(ref))


def checks(rows):
    """Pass/fail lines for the step 2c thresholds, from the rows of evaluate()."""
    out = []
    by_case = {r["case"]: r for r in rows}

    def add(label, value, ok):
        out.append((label, value, ok))

    t = by_case["tremolo"]
    add("tremolo amp correlation >= 0.8", t["amp_corr"], t["amp_corr"] >= 0.8)
    add("tremolo rate within 5% of 6 Hz", t["amp_rate_hz"], abs(t["amp_rate_hz"] - 6.0) <= 0.3)
    p = by_case["pulse"]
    add("pulse amp correlation >= 0.8", p["amp_corr"], p["amp_corr"] >= 0.8)
    add("pulse rate within 5% of 4 Hz", p["amp_rate_hz"], abs(p["amp_rate_hz"] - 4.0) <= 0.2)

    v = by_case["vibrato"]
    add("vibrato rate within 5% of 5 Hz", v["vib_rate_hz"], abs(v["vib_rate_hz"] - 5.0) <= 0.25)
    ratio = v["vib_depth_cents"] / max(v["vib_depth_ref_cents"], 1e-9)
    add("vibrato depth within 30% of the reference", ratio, abs(ratio - 1.0) <= 0.3)

    s = by_case["sweep"]
    add("brightness correlation >= 0.6", s["bright_corr"], s["bright_corr"] >= 0.6)

    static_ok = all(r["static_amp_corr"] < 0.5 for r in rows if r["case"] in ("tremolo", "pulse"))
    add("movement 0 is clearly worse (amp correlation < 0.5)",
        {r["case"]: round(r["static_amp_corr"], 3) for r in rows if r["case"] in ("tremolo", "pulse")},
        static_ok)
    if "v2_amp_corr" in t:
        v2_ok = all(by_case[c]["v2_amp_corr"] < by_case[c]["amp_corr"] - 0.2 for c in ("tremolo", "pulse"))
        add("v2 export is clearly worse on amplitude (by 0.2+)",
            {c: round(by_case[c]["v2_amp_corr"], 3) for c in ("tremolo", "pulse")}, v2_ok)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
                        help="step 2b model for the v2 baseline")
    args = parser.parse_args(argv)
    rows = evaluate(args.checkpoint)
    for row in rows:
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()})
    failed = 0
    for label, value, ok in checks(rows):
        failed += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {label}: {value}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
