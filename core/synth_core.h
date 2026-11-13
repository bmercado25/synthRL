// Portable DSP core. No Apple dependencies. Used by the AUv3 wrapper, the
// offline test renderer and (later) the Python side.
#ifndef SYNTH_CORE_H
#define SYNTH_CORE_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SynthCore SynthCore;

// Allocates the core. Not real-time safe; call off the audio thread.
SynthCore* synth_create(double sample_rate);
void synth_destroy(SynthCore* core);

// MIDI-style note control. Velocity 0-127.
void synth_note_on(SynthCore* core, int note, int velocity);
void synth_note_off(SynthCore* core, int note);

// Renders `frames` mono samples into `out`. No allocation, no locks.
void synth_render(SynthCore* core, float* out, int frames);

#ifdef __cplusplus
}
#endif

#endif  // SYNTH_CORE_H
