# Generative Synthesizer: Project Context

An Apple AUv3 subtractive synthesizer, built from scratch, designed so a reinforcement learning system can later discover presets on top of it. The RL system finds presets that match a goal (a reference sound or a text description) and keeps the results diverse.

Read this file before making architectural decisions. The full plan lives in the project document "AI-Driven Generative Synthesizer: Project Plan & Technical Architecture".

## Core idea

An autonomous sound-design system that learns to operate its own synthesizer.

1. A goal is encoded as a vector `g` by a frozen, pretrained audio-text embedding model (e.g. CLAP). Audio and text goals share one space.
2. A goal-conditioned SAC agent refines a patch over 10 to 20 steps per episode.
3. Many episodes run in parallel. Final patches go into a discovery archive.
4. A novelty bonus (gated on goal similarity) keeps discoveries distinct.
5. The plugin itself plots the archive as an exploration map for the user to audition. There is no separate companion app; everything runs inside the same plugin.

## Architecture rules (these keep the RL layer possible)

1. **Portable DSP core.** All DSP is a C++ library with no Apple dependencies. The AUv3 is a thin wrapper. The same core runs in the plugin, an offline renderer and a Python module.
2. **Parameters declared once.** A single schema holds each parameter's ID, name, range, default, normalized 0-1 mapping, and continuous/discrete type. The `AUParameterTree` is generated from it, and it is exported as JSON for the Python side. Never hand-declare a parameter in two places.
3. **Deterministic offline rendering.** `render(patch, midi_phrase, sample_rate, seconds) -> audio`, faster than real time, with no dependence on the audio thread or a host. All randomness (noise, LFO phase) is seeded. The same patch must always render the same audio.
4. **Patch = flat parameter vector + schema version.** Presets save and load through the AU's full state, so discovered patches open directly in the plugin.
5. **Modular internals.** Build the signal path as modules with routable connections, even if the first UI is a conventional synth panel.
6. **Nothing learned in the audio thread.** Training and candidate evaluation run out of process. The plugin only loads and plays presets.
7. **Training does not host the AUv3.** AUv3 extensions are sandboxed and awkward to drive in bulk. Training uses the portable core directly.
8. **Real-time safety.** No allocation, locks, file I/O or logging in the render callback.

## Synth v1 scope

- Two oscillators (waveform selection, detune), mixer
- Low-pass filter with resonance
- ADSR envelopes, LFO with modulation depth
- Reverb and basic effects
- MIDI input

Keep the parameter space small. Prefer continuous controls: for example, a saw-to-square blend instead of a hard waveform switch, because discrete actions are awkward for SAC.

## RL summary

- **Algorithm:** Soft Actor-Critic, run as iterative refinement. Each step applies bounded adjustments to normalized parameters, clipped to 0-1.
- **Observation:** `s = [g, p, e, f, z]` (goal embedding, current parameters, embedding of current audio, cheap DSP features, optional per-episode style vector).
- **Per-step reward:** improvement in cosine similarity between the rendered audio embedding and the goal.
- **Final reward:** similarity + gated novelty bonus - penalties (silence, clipping, DC offset, runaway resonance).
- **Novelty:** mean distance to the k nearest neighbours (k around 5-15) in the archive, in embedding space. Never in parameter space.
- **Hindsight relabeling:** substitute the embedding of the sound actually produced as the goal, so failed attempts still teach the parameter-to-sound mapping.
- **Diversity mechanisms, in order:** novelty bonus, latent style vector `z`, MAP-Elites archive over interpretable descriptors (brightness, attack time, modulation rate). Investigate PGA-MAP-Elites.
- **Embedding model stays frozen.** Archive distances are only meaningful if embeddings are computed identically every time.
- **Fixed test phrases** (held chord, slow melody, staccato line) are used for every evaluation, since perceived character depends on the notes played.

## Baselines the RL must beat

Random search, CMA-ES, and a supervised inverse model trained on (parameters, embedding) pairs. If RL does not beat these on the same goals and render budget, it is not yet justified. A supervised model may also warm-start the policy.

## Roadmap

1. Portable DSP core, parameter schema, deterministic offline renderer, Python binding
2. AUv3 wrapper (parameter tree from schema, MIDI, presets); verify offline render matches the plugin
3. Audio analysis: test phrases, DSP features, frozen embeddings, throughput measurement
4. Baselines: random search, CMA-ES, supervised inverse model
5. Goal-conditioned SAC with hindsight relabeling on audio-matching goals
6. Diversity: archive, novelty bonus, latent `z`, MAP-Elites, parallel environments
7. Plugin UI: exploration map, auditioning, comparison, preset export (in the plugin, not a separate app)
8. Text goals and human preference learning
9. Modular synthesis expansion

Each phase ends with something testable. Do not start a later phase to avoid a failing earlier one.

## Proposed stack

| Component | Technology |
| --- | --- |
| DSP core | C++ |
| Plugin | AUv3 Audio Unit instrument |
| Plugin UI | SwiftUI / AppKit |
| Python access to engine | pybind11 or headless CLI renderer |
| RL | PyTorch, SAC |
| Parallel environments | Python multiprocessing initially |
| Audio analysis | librosa / custom DSP |
| Embeddings | Pretrained audio-text model (e.g. CLAP), frozen |

## Open questions

1. Does goal-conditioned SAC with hindsight relabeling beat the baselines on audio-matching goals?
2. Does the embedding model's text space align with how people describe synth sounds?
3. Does a gated novelty bonus produce a spread-out archive, or is a latent `z` needed?
4. How many renders and embeddings per second can the pipeline sustain?
5. How well do automated scores agree with human ratings?

## Conventions (fill in as decisions are made)

- Repository layout:
- Build system:
- Minimum macOS / iOS targets:
- Parameter schema file location:
- Test phrase files location:
- Python environment:

## Decision log

| Date | Decision | Reason |
| --- | --- | --- |
| 2026-11-12 | Build the AUv3 synth from scratch around a portable C++ core | Lets the same DSP run in the plugin, an offline renderer and Python for RL training |
| 2026-11-12 | Goals are embedding vectors, not hand-picked emotion dimensions | Richer, learned space; audio and text goals share one policy |
| 2026-11-12 | Iterative-refinement SAC with gated novelty bonus | Inspectable steps; diversity measured in sound space |
| 2026-11-12 | No companion app; exploration map, auditioning, comparison and preset export live inside the plugin | One codebase and one deliverable; the plugin already holds the core, the presets and the audio |
