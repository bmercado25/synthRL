#include "synth_core.h"

#include <cmath>

// Rudimentary v0: one monophonic sine voice with a short linear fade in and
// out. Enough to prove the MIDI -> audio path end to end.
struct SynthCore {
    double sampleRate;
    double phase = 0.0;        // radians
    double phaseInc = 0.0;     // radians per sample
    double amp = 0.0;          // current amplitude
    double ampTarget = 0.0;    // 0 or the note's velocity amplitude
    double ampStep = 0.0;      // amplitude change per sample during a fade
    int    activeNote = -1;
};

static const double kFadeSeconds = 0.005;
static const double kMaxAmplitude = 0.5;

static double midi_note_to_hz(int note) {
    return 440.0 * std::pow(2.0, (note - 69) / 12.0);
}

SynthCore* synth_create(double sample_rate) {
    SynthCore* core = new SynthCore();
    core->sampleRate = sample_rate;
    return core;
}

void synth_destroy(SynthCore* core) {
    delete core;
}

void synth_note_on(SynthCore* core, int note, int velocity) {
    if (velocity <= 0) {
        synth_note_off(core, note);
        return;
    }
    const double twoPi = 6.283185307179586;
    core->activeNote = note;
    core->phaseInc = twoPi * midi_note_to_hz(note) / core->sampleRate;
    core->ampTarget = kMaxAmplitude * (velocity / 127.0);
    core->ampStep = core->ampTarget / (kFadeSeconds * core->sampleRate);
}

void synth_note_off(SynthCore* core, int note) {
    if (note != core->activeNote) {
        return;
    }
    core->activeNote = -1;
    core->ampTarget = 0.0;
    core->ampStep = core->amp / (kFadeSeconds * core->sampleRate);
}

void synth_render(SynthCore* core, float* out, int frames) {
    const double twoPi = 6.283185307179586;
    for (int i = 0; i < frames; ++i) {
        if (core->amp < core->ampTarget) {
            core->amp += core->ampStep;
            if (core->amp > core->ampTarget) core->amp = core->ampTarget;
        } else if (core->amp > core->ampTarget) {
            core->amp -= core->ampStep;
            if (core->amp < core->ampTarget) core->amp = core->ampTarget;
        }
        out[i] = static_cast<float>(core->amp * std::sin(core->phase));
        core->phase += core->phaseInc;
        if (core->phase >= twoPi) core->phase -= twoPi;
    }
}
