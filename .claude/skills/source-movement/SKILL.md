---
name: source-movement
description: Step 2c of Neural Synth. Measure an imported sample's loudness, pitch and brightness movement directly from the audio, export it in timbre format v3, and play it back looped in the C++ core behind a Movement control. Use when working on making imported sounds keep the reference's character or cadence, or on the movement tracks, the v3 timbre format, or the Movement parameter.
---

# Step 2c: carry the source's movement into the synth

Read CLAUDE.md first. This step follows the retrained 2b model (best checkpoint
`data/checkpoints/nsynth_2b_full.pt`, validation 2.23).

## Why
The model captures timbre, but imported sounds lose the reference's character and
cadence. Causes: the encoder mean-pools over time (`Encoder.forward`,
`neural/model.py`); the export is one static harmonic shape plus a loudness curve
the decoder predicts (not measured) over one 4 s window (`export_from_model`,
`fit_to_note`); the core played one envelope once and had one unrelated LFO.

Decisions made with the user (do not re-ask): all of amplitude pulsing, pitch
wobble, evolving brightness and phrase rhythm matter; movement loops while a key
is held; analysis only, no retrain and no model change; the decoder keeps supplying
the harmonic shape.

## Design (as built)
Three tracks at 50 Hz (20 ms), up to 10 s (500 frames), plus a loop region.
- `amp`: smoothed RMS, linear, peak 1.
- `pitch_cents`: pYIN contour in cents relative to the median, clipped to +-600;
  zeros if unpitched. Unpitched frames inside a pitched clip are interpolated.
- `brightness_oct`: spectral centroid in octaves below the clip's 90th percentile
  of audible frames, clipped to [-4, 0]. The filter is open by default and can only
  close, so the brightest part of the source sits at the user's cutoff.
- Loop region: starts at the loudness peak, ends at the pair (start, end) with the
  smallest normalized jump across the three tracks, minimum 1 s. A 100 ms crossfade
  at the wrap blends the tail toward the frames just before the loop start. Clips
  under 1 s loop whole.

Core (`core/synth_core.cpp`), per voice, audio thread, no allocation:
- Frame position = `envSamples * 50 / sampleRate`. Play from frame 0, wrap inside
  `[loopStart, loopEnd)`, linearly interpolate.
- amp: `gain = 1 + movement * (amp - 1)` on the neural source.
- pitch: neural source `cps * 2^(movement * cents / 1200)`, on the neural source only.
- brightness: cutoff offset `sourceMix * movement * brightness_oct` octaves.
- `SYNTH_PARAM_MOVEMENT` (address 24, default 1). With movement tracks present, the
  envelope is ignored.

## Files
- `neural/movement.py`: `measure_movement(wave_16k)`, `Movement`, the loop search.
  Pitch uses `PITCH_FRAME = 1024`: a 2048 window halves a 5 Hz vibrato's depth.
- `neural/model.py`: `loudest_window(x, n, hop)`, factored out of `fit_to_note`.
- `neural/timbre.py`: version 3 with an optional `movement` block; `read_movement`.
  Versions 1 and 2 still load. Writing both an envelope and movement raises.
- `neural/export.py`: writes movement in analysis and model modes; `--no-movement`.
- `neural/coreplay.py`: drives `build/libsynthcore.dylib` through ctypes. Parameter
  indices are read from `schema/synth_params.json`, never hard-coded.
- `neural/movement_eval.py`: the objective check through the real core.
- `core/synth_core.h/.cpp`: `SYNTH_PARAM_MOVEMENT`, `SynthMovement`,
  `synth_set_timbre_moving`; tracks live in `TimbreSet`.
- `plugin/SynthAudioUnit.swift`: `TimbreMovement`, the kernel path, and the
  `synthTimbreMovement` fullState key.
- `host/SynthHostApp.swift`: `parseMovement`; `apply` writes or removes the key.
- `ui/SynthUI.swift`: a `source` panel with the `move` knob.

## Verify
```
./scripts/test.sh                              # core tests, schema, Python, AUv3
                                               # + full pipeline: tremolo clip -> neural.export
                                               #   -> AU loads the v3 file, checks it is audible,
                                               #   that loudness moves at Movement 1 and holds at 0,
                                               #   and that the fullState route (host path) works
.venv/bin/python -m neural.movement_eval       # objective check through the core
```
Last results (2026-11-22): every threshold passes. Tremolo amplitude correlation
0.996 at 5.88 Hz (6 Hz). Pulse correlation 0.996 at 3.92 Hz (4 Hz). Vibrato rate
4.97 Hz (5 Hz), depth 0.87 of the reference. Brightness sweep correlation 0.618,
which passes the 0.6 bar narrowly. Movement 0 gives amplitude correlation near zero.
The v2 export gives near zero too.

Listening WAVs are in `build/movement_listen/` (source, movement0, movement1) for
three NSynth clips. They were written, not yet judged by ear.

## Known limits (state them in results)
- One static harmonic shape; spectral movement is carried only as brightness.
- Unpitched sounds get no pitch track.
- Clips over 10 s are cropped to their loudest 10 s window.
- Tracks are pitch-independent.
- A noise-like source loops its measured amplitude, so it repeats.
- The brightness sweep sits near the 0.6 bar. The filter is a single SVF, so it
  does not reproduce every spectral shape.
- NSynth notes are steady, so their movement is small. Use the user's own clips to judge.
