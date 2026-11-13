// Portable DSP core. No Apple dependencies. Used by the AUv3 wrapper, the
// offline test renderer and (later) the Python side.
//
// The parameter schema lives here and is the single source of truth. The AU
// builds its AUParameterTree from synth_param_info(), and the host UI reads
// the same table.
#ifndef SYNTH_CORE_H
#define SYNTH_CORE_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SynthCore SynthCore;

// ---- Parameters ------------------------------------------------------------
// Index order is part of the contract: it is the AU parameter address.
enum {
    SYNTH_PARAM_OSC_A_WAVE = 0,
    SYNTH_PARAM_OSC_A_LEVEL,
    SYNTH_PARAM_OSC_B_WAVE,
    SYNTH_PARAM_OSC_B_LEVEL,
    SYNTH_PARAM_FILTER_CUTOFF,     // normalized 0-1, mapped exponentially to 20 Hz - 20 kHz
    SYNTH_PARAM_FILTER_RESONANCE,  // normalized 0-1: 0 = flat (no peak), 1 = strong peak
    SYNTH_PARAM_ENV_ATTACK,        // normalized 0-1, time mapped exponentially 1 ms - 10 s
    SYNTH_PARAM_ENV_DECAY,         // normalized 0-1, time mapped exponentially 1 ms - 10 s
    SYNTH_PARAM_ENV_SUSTAIN,       // level 0-1 (linear)
    SYNTH_PARAM_ENV_RELEASE,       // normalized 0-1, time mapped exponentially 1 ms - 10 s
    SYNTH_PARAM_COUNT
};

// Kind of parameter. Waveform parameters are integers 0..waveform_count-1.
enum {
    SYNTH_PARAM_CONTINUOUS = 0,
    SYNTH_PARAM_WAVEFORM = 1
};

// Waveform choices. The same list is offered by both oscillators.
enum {
    SYNTH_WAVE_SINE = 0,
    SYNTH_WAVE_SAW,
    SYNTH_WAVE_SQUARE,
    SYNTH_WAVE_TRIANGLE,
    SYNTH_WAVE_COUNT
};

typedef struct {
    const char* id;          // stable identifier, e.g. "oscA.wave"
    const char* name;        // display name
    int kind;                // SYNTH_PARAM_CONTINUOUS or SYNTH_PARAM_WAVEFORM
    float min;
    float max;
    float defaultValue;
} SynthParamInfo;

int synth_param_count(void);
SynthParamInfo synth_param_info(int index);
int synth_waveform_count(void);
const char* synth_waveform_name(int index);

// Sets a parameter. Values are clamped to the parameter's range. Safe to call
// from any thread while the core renders.
void synth_set_param(SynthCore* core, int index, float value);
float synth_get_param(SynthCore* core, int index);

// Converts a normalized filter cutoff (0-1) to Hz. Exponential, so equal knob
// steps are equal musical intervals: 0 -> 20 Hz, 1 -> 20 kHz.
float synth_cutoff_hz(float normalized);

// Converts a normalized envelope time (0-1) to seconds. Exponential: 0 -> 1 ms,
// 1 -> 10 s, so the knob is fine-grained at short times where it matters most.
float synth_envelope_seconds(float normalized);

// ---- Lifecycle and performance --------------------------------------------
// Allocates the core. Not real-time safe; call off the audio thread.
SynthCore* synth_create(double sample_rate);
void synth_destroy(SynthCore* core);
void synth_set_sample_rate(SynthCore* core, double sample_rate);

// MIDI-style note control. Velocity 0-127.
void synth_note_on(SynthCore* core, int note, int velocity);
void synth_note_off(SynthCore* core, int note);

// Renders `frames` mono samples into `out`. No allocation, no locks.
void synth_render(SynthCore* core, float* out, int frames);

#ifdef __cplusplus
}
#endif

#endif  // SYNTH_CORE_H
