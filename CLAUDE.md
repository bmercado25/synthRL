# Neural Synth: Project Context

An AI-powered synthesizer plugin built as an AUv3 Audio Unit on macOS, with a SwiftUI host app. The DSP lives in a portable C++ core. A neural audio generator creates the underlying waveform from an imported sample, and a reinforcement learning agent explores new sounds live while the user plays. The user watches, auditions and guides the exploration, and every discovery can be saved as a reproducible preset.

Read this file before making architectural decisions. The full plan lives in the project document "Neural Synth: Product Requirements & Technical Architecture".

## Core idea

An autonomous AI sound designer operating inside a synthesizer.

1. The user imports an audio sample (WAV first). The analysis engine extracts a learned representation `z_src` that keeps both temporal and spectral information.
2. A reusable neural generator, conditioned on `z_src`, renders playable audio for any MIDI pitch and velocity. It is never retrained per sample and never just resamples the source.
3. An exploration agent runs continuously in the background after the user presses EXPLORE. Each cycle: choose an action, build a candidate, render it, analyze it, compute the reward, publish it for auditioning, update state.
4. The user plays and auditions while exploration continues. KEEP, DISCARD, FAVORITE, MORE LIKE THIS and LESS LIKE THIS feed back into the agent.
5. Candidates form an exploration tree (original sample at the root). Any node can be auditioned, compared with its parent, branched from, or saved as a preset.
6. Traditional synth controls (filter, ADSR, LFO, modulation matrix, effects, AI MORPH) sit on top of the neural source. Everything runs inside the same plugin; there is no separate companion app.

## Architecture rules

1. **Three independent systems.** Audio thread (playback, MIDI, filters, envelopes, modulation, effects), neural generation worker (inference, buffers, pre/post-processing), RL exploration worker (actions, evaluation, policy updates, history, feedback). The audio thread never waits on inference or training. Never run expensive inference or RL training in the audio callback.
2. **Safe hand-off.** Thread-safe queues and double-buffering move generated audio to the playback engine. Candidate switching is click-free (crossfade). Candidate generation never blocks the UI.
3. **Reusable conditioned generator.** One neural generator, conditioned on the source embedding. Pitch and velocity are inputs, so the instrument plays across a useful range. Latent representations support continuous manipulation (needed for AI MORPH).
4. **Replaceable parts.** The neural generator, exploration algorithm, reward system, DSP engine and UI are separate modules. Swapping the exploration algorithm must not require changes to the generator.
5. **Parameters declared once.** A single schema holds each parameter's ID, name, range, default, normalized 0-1 mapping and continuous/discrete type, for both the traditional controls and the generator conditioning parameters. Host parameters are generated from it and it is exported as JSON for the Python side. Never hand-declare a parameter in two places.
6. **Portable DSP core.** Filters, envelopes, LFOs, modulation matrix, waveform manipulation and effects live in a C++ library with no plugin-framework dependency, so the plugin, a standalone prototype and offline tools share it.
7. **Candidate = recipe, not audio.** A candidate stores generator version, conditioning data, latent vector, synthesis parameters, RL action history, reward components, parent ID, feedback and timestamp. Presets save the recipe, not just a waveform.
8. **Reproducibility.** Seed all randomness and pin the generator version. Exact neural regeneration is not guaranteed, so presets may carry an optional cached audio representation as a fallback.
9. **User controls are independent of exploration.** The user can edit the current candidate while the agent keeps exploring. PAUSE preserves exploration state. STOP is immediate and never interrupts playback.
10. **Honest synthesis graph.** The graph shows the real processing chain (for example Neural Source → Harmonic Processor → Wavefolder → Filter → Amplifier). Do not invent an oscillator architecture for a neural source.
11. **Visualization is a projection.** The latent-space view (UMAP or PCA) is not the true geometry of the model's latent space. Label it that way.
12. **Real-time safety.** No allocation, locks, file I/O or logging in the render callback. Bounded memory. Safe parameter updates.
13. **Resource fallbacks.** Handle missing GPU gracefully. Provide a fallback mode for machines that cannot run the model at the preferred speed.
14. **Deployment.** The first prototype may use a separate local inference process. The shipped plugin must not need a permanently running Python environment.

## Synth scope

- **Neural oscillator section:** generated waveform display, latent-space position, neural variation, harmonic complexity, waveform morphing, phase manipulation, optional folding
- **Waveform manipulation:** wavetable-style scanning where supported, wavefolding, bit reduction, FM, AM, phase distortion, nonlinear distortion (processing operations, not oscillator sources)
- **Filter:** low-pass, high-pass, band-pass, notch; cutoff, resonance, envelope amount, drive
- **Envelopes:** ADSR for amplitude and filter
- **LFO:** rate, depth, waveform, phase, sync, destination
- **Modulation matrix:** latent variables, harmonic complexity, cutoff, resonance, distortion, amplitude, pitch, other supported parameters
- **Effects:** reverb, delay, chorus, saturation, stereo widening, compression
- **AI MORPH:** 0% original-inspired, 50% hybrid, 100% highly transformed. It operates on the learned representation or generator conditioning, not by crossfading unrelated audio. Interpolation must keep perceptual continuity.

Build the parameter space incrementally. Prefer continuous controls over discrete switches.

## Audio analysis

Extract f0 and pitch, harmonic structure, spectral centroid, bandwidth, rolloff and flux, zero-crossing rate, transients, ADSR characteristics, temporal envelope, noise characteristics, inharmonicity, MFCCs, and embeddings from a learned audio encoder. Do not collapse the sample to a single vector if that discards temporal information.

## Neural generator

Candidates to investigate: DDSP, neural waveform generators, neural vocoders, conditional generative audio models, latent-variable audio models, neural spectral synthesis. Inputs: source embedding, current latent, MIDI pitch, velocity, note duration, generation parameters, modulation parameters, optional exploration conditioning. Investigate existing research and implementations before building a new model from scratch.

## RL and exploration summary

- **Policy:** contextual, `π(a | s, z)`, where `s` is the exploration state and `z` the encoded source audio.
- **State:** current candidate parameters, generated audio features, previous scores, exploration history, user feedback, source embedding.
- **Actions:** modify latents, harmonic complexity, brightness, transients, distortion, noise, FM, spectral envelope, temporal behavior; try other configurations; mutate promising candidates; vary saved candidates.
- **Algorithms to compare:** PPO, SAC, contextual bandits, evolutionary strategies. Judge on stability, compute cost, exploration quality and ability to use human feedback. Do not assume PPO.
- **Reward:** `R = w1·S + w2·N + w3·P + w4·M + w5·U` (similarity, novelty, playability, perceptual quality, user preference). Each term is independently configurable.
  - *S:* spectral, mel-spectrogram, harmonic, transient and embedding similarity.
  - *N:* distance to existing candidates. Never reward noise or distortion just for being different.
  - *P:* pitch consistency, velocity response, transients, note transitions, stability across the MIDI range, artifacts.
  - *U:* contextual feedback. A rejected candidate is not universally bad. Preference can persist across sessions only if the user enables personalization.
- **Embedding model stays frozen** wherever embeddings are compared, so distances stay meaningful.
- **Fixed test phrases** (held chord, slow melody, staccato line) are used for every evaluation.
- **Personalities** are reward weightings or policy-conditioning parameters, not separate generators: Faithful, Experimental, Harmonic, Organic, Evolution (population plus mutation and branching), Human-Guided.
- **Live controls:** Novelty, Similarity, Exploration Speed (scheduling and throughput only, never audio sample rate), plus depth, mutation strength, population size, retention, seed, temperature, feedback influence. Transport: START EXPLORING, PAUSE, RESUME, STOP, RESET, SAVE DISCOVERY.

## Baselines the RL must beat

A lightweight evolutionary or Bayesian explorer comes first, before any RL. Also compare random search, CMA-ES and a supervised inverse model trained on (parameters, embedding) pairs. If RL does not beat these on the same goals and render budget, it is not yet justified.

## Visualization

Waveform view (original vs generated), spectrogram view, latent-space view, synthesis graph, exploration tree, reward components, current policy actions, candidate history.

## Roadmap

1. Neural sound generation prototype: sample import, analysis, conditional generator, pitch and velocity conditioning, playback, simple preset
2. Live exploration: candidate parameter space, background worker, queues, waveform display, auditioning, A/B, save/discard, baseline search
3. Reward and evaluation: similarity, novelty, playability, ranking, user feedback, history
4. Reinforcement learning: contextual policy, observation and action spaces, personalities, branching, comparison with baselines
5. Synthesizer interface: filters, ADSR, LFOs, mod matrix, waveform manipulation, effects, AI MORPH, MIDI, presets
6. Exploration visualization: waveform, spectrogram, latent map, tree, reward and action displays
7. Plugin integration and optimization: AUv3 wrapper (AUAudioUnit) around the C++ core, thread-safe model communication, inference optimization, preset I/O, MIDI validation, audio-thread stability, long-run tests, CPU and memory profiling

First establish three things before expanding: a neural generator that produces playable conditioned audio, a background exploration engine that never interrupts playback, and a feedback mechanism that guides exploration.

Each phase ends with something testable. Do not start a later phase to avoid a failing earlier one.

## Proposed stack

| Component | Technology |
| --- | --- |
| Plugin and DSP | C++ portable core, Swift AUAudioUnit wrapper, AUv3 Audio Unit (macOS) |
| Host app and UI | SwiftUI with AppKit where needed (AU view controller via CoreAudioKit) |
| Neural generation | PyTorch, DDSP, neural audio models, audio embedding models |
| Audio analysis | librosa, torchaudio, custom C++ DSP analysis |
| RL | Stable-Baselines3 (initial experiments), PPO, SAC, custom contextual policies, evolutionary baselines |
| Visualization | SwiftUI / AppKit views, GPU (Metal) waveform rendering where needed, custom spectrogram, graph view for the tree |
| Prototype inference | Separate local process (shipped plugin must not depend on Python) |

## Open questions

1. Which generator gives the best mix of waveform quality, controllability and inference speed?
2. Can it play across a wide MIDI range? *Partial (step 0 spike, random-weight DDSP-style harmonic decoder, `spikes/step0_render_speed.py`): all 88 notes (MIDI 21-108) render with the spectral peak within 50 cents of a harmonic of f0. This checks the harmonic path only, not a trained model or perceived pitch.*
3. Which representations support meaningful latent manipulation?
4. Is generation fast enough for continuous background exploration? *Partial (same spike): the tiny decoder runs at about 0.001-0.004 of real time per voice on CPU and MPS. That is far below a realistic neural vocoder's cost, so it is an optimistic floor, not a verdict.*
5. Which RL algorithm works best for continuous synthesis parameters, and does it beat the baselines?
6. How should human preference feedback be incorporated?
7. How does exploration stay stable while the user changes parameters?
8. Can presets be reconstructed reliably without storing audio?
9. What is the best strategy for CPU-only inference?
10. Can several candidate-generation workers run at once?
11. How well do automated scores agree with human ratings?

## Conventions (fill in as decisions are made)

- Repository layout: `core/` (portable C++ DSP, `synth_core.h/.cpp`), `plugin/` (AUv3 wrapper, bridging header, Info.plist, entitlements), `ui/` (SwiftUI controls), `host/` (SwiftUI host app), `tests/` (`core_test.cpp`, `au/main.swift`), `scripts/` (build, install, test)
- Build system: shell scripts over `clang++` and `swiftc` (Command Line Tools, no full Xcode required). `scripts/build.sh` produces `build/SynthHost.app` with the AUv3 extension embedded and ad-hoc signed
- Minimum OS targets:
- Parameter schema file location: `schema/synth_params.json`, generated from the core by `tools/export_schema.cpp`. `scripts/test.sh` fails if it is stale
- Test phrase files location:
- Python environment: Python 3.12 in `.venv/` at the repo root (`uv venv --python 3.12 .venv`), pinned in `spikes/requirements.lock`. PyTorch 2.15 with MPS (Apple GPU) available. The venv is git-ignored
- Movement (step 2c): timbre format version 3 carries an optional `movement` block (loudness, pitch and brightness tracks at 50 Hz, max 500 frames, plus a loop region). Measured from the audio by `neural/movement.py`, played by the core behind `SYNTH_PARAM_MOVEMENT` (AU address 24). Movement replaces the envelope when present
- Attack spectrum (step 2d): timbre version 5 adds `harmonics_attack`, the first 100 ms on the sustain's scale. Core parameters Spectral Amount (index 26) and Spectral Time (index 27) blend toward it after note-on. Default amount 0 leaves the sound unchanged (checked by the core test and a golden render). Filter Mode (index 25) blends LP, BP, HP and notch. Both are off in the default fit
- Patch (step 2d): timbre format version 4 adds an optional `patch` block, a map from schema id to `{index, value}`, checked against the schema on write and read. `neural/fit.py` holds the shared similarity loss and the staged CMA-ES fit (`FIT_PARAMS`, time-capped). `python -m neural.export clip.wav out.json --fit` writes it. The host applies the patch through the parameter tree and an A/B button swaps source and fit
- Python renders through the real core: `neural/coreplay.py` drives `build/libsynthcore.dylib` (built by `scripts/test.sh`). Objective movement check: `python -m neural.movement_eval`
- Generator model and version:

## Decision log

| Date | Decision | Reason |
| --- | --- | --- |
| 2026-11-12 | Build the synth around a portable C++ core | Lets the same DSP run in the plugin, a standalone prototype and offline tools |
| 2026-11-12 | Goals are embedding vectors | Richer, learned space; source audio conditions one shared policy |
| 2026-11-12 | No companion app; map, auditioning, comparison and export live inside the plugin | One codebase and one deliverable |
| 2026-11-12 | Iterative-refinement SAC with gated novelty bonus | **Superseded 2026-11-16:** SAC is now one of several algorithms to compare, behind an evolutionary baseline |
| 2026-11-16 | Sound source is a neural generator, not classic oscillators | Core product idea from the Neural Synth spec; classic DSP kept for filter, envelopes, modulation and effects |
| 2026-11-16 | Exploration runs live inside the plugin on background workers, off the audio thread | Defining feature of the product. Replaces the earlier "nothing learned in the plugin" rule; the audio thread still never runs inference or training |
| 2026-11-16 | Target VST3 and AU with JUCE | **Superseded 2026-11-16 by the AUv3 decision below** |
| 2026-11-16 | Shell is AUv3 (Swift AUAudioUnit wrapper, SwiftUI host app) over the portable C++ core; no JUCE, no VST3 for now | Matches the existing project code (core/, plugin/, host/, ui/). Keeps the DSP portable so a VST3 or JUCE shell can be added later without rewriting it |
| 2026-11-16 | Human feedback is part of the core loop, not a late phase | KEEP, DISCARD, FAVORITE, MORE/LESS LIKE THIS guide exploration from the start |
| 2026-11-16 | Presets store the recipe, with optional cached audio | Neural regeneration may not be exactly reproducible |
| 2026-11-16 | Step 2a is analysis-resynthesis (pYIN f0, harmonic timbre, no training); the trained generator comes after the hand-off works | Proves sample import, the core's neural source path and the host loader end to end before model quality is at stake. Known limit: magnitudes only, no phase or noise |
| 2026-11-22 | Step 2c carries the source's movement (loudness, pitch, brightness and a looped phrase) into the synth, measured from the audio and played in the core behind a Movement control. Analysis only, no retrain | The decoder's envelope was predicted, not measured, and the encoder pools over time, so imported sounds lost their cadence and wobble. Movement loops while a key is held. Pitch is measured with a 1024-sample pYIN window, since 2048 halves a 5 Hz vibrato's depth |
| 2026-11-22 | Step 2d fast fit: staged CMA-ES over the synth controls, time-capped, lowers the loss by a median 13-21% on 8 real notes and a held-out distance by 2-6%. The fitted patch ships in timbre v4 so the host can A/B it | Self-recovery on random patches missed its gate, but the fit helps real notes. The controls cannot reproduce the spectrum the neural source misses, so new tools (noise, transients, filter modes) are the next step, not more optimizer work |
| 2026-11-22 | Step 2d tools, measured and not kept by default: filter mode (LP/BP/HP/notch), an attack spectrum with spectral amount and time (timbre v5), and a 4 kHz pYIN range. The fit stays on the 10 core controls | Ablation on 24 validation notes, same budget and seeds: no added group cut the fitted loss by more than noise (median +0.0 to +0.1%). Filter mode and spectral amount never moved from their start in any note, so the optimizer found no use for them. The residual error (static offset and time-varying level, about 5 dB each) is not fixed by these controls. The octave guard was removed: it made pitch worse on labelled notes, and raising the pYIN range fixed the octave error |
