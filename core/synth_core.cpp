#include "synth_core.h"

#include <atomic>
#include <cmath>

namespace {

const double kTwoPi = 6.283185307179586;
const double kMasterAmplitude = 0.5;

const SynthParamInfo kParams[SYNTH_PARAM_COUNT] = {
    {"oscA_wave",  "Osc A Waveform", SYNTH_PARAM_WAVEFORM,
     0.0f, SYNTH_WAVE_COUNT - 1.0f, SYNTH_WAVE_SINE},
    {"oscA_level", "Osc A Level",    SYNTH_PARAM_CONTINUOUS, 0.0f, 1.0f, 1.0f},
    {"oscB_wave",  "Osc B Waveform", SYNTH_PARAM_WAVEFORM,
     0.0f, SYNTH_WAVE_COUNT - 1.0f, SYNTH_WAVE_SINE},
    {"oscB_level", "Osc B Level",    SYNTH_PARAM_CONTINUOUS, 0.0f, 1.0f, 0.0f},
    {"filter_cutoff",    "Filter Cutoff",    SYNTH_PARAM_CONTINUOUS, 0.0f, 1.0f, 1.0f},
    {"filter_resonance", "Filter Resonance", SYNTH_PARAM_CONTINUOUS, 0.0f, 1.0f, 0.0f},
    // Defaults: attack 5 ms, decay 100 ms, sustain full, release 50 ms.
    {"env_attack",  "Attack",  SYNTH_PARAM_CONTINUOUS, 0.0f, 1.0f, 0.1748f},
    {"env_decay",   "Decay",   SYNTH_PARAM_CONTINUOUS, 0.0f, 1.0f, 0.5f},
    {"env_sustain", "Sustain", SYNTH_PARAM_CONTINUOUS, 0.0f, 1.0f, 1.0f},
    {"env_release", "Release", SYNTH_PARAM_CONTINUOUS, 0.0f, 1.0f, 0.4257f},
};

const double kMinCutoffHz = 20.0;
const double kMaxCutoffHz = 20000.0;

const double kMinEnvSeconds = 0.001;
const double kMaxEnvSeconds = 10.0;

// Resonance maps to the SVF damping term k (k = 1/Q). k = sqrt(2) is a flat,
// Butterworth response. Lower k gives a peak at the cutoff. k is kept at
// 0.1 or above (Q up to 10) so the filter stays well behaved.
const double kFlatDamping = 1.4142135623730951;
const double kMinDamping = 0.1;

const char* const kWaveNames[SYNTH_WAVE_COUNT] = {"Sine", "Saw", "Square", "Triangle"};

enum EnvStage { kEnvIdle, kEnvAttack, kEnvDecay, kEnvSustain, kEnvRelease };

// PolyBLEP residual: removes the aliasing from a waveform's sharp step.
// `t` is phase in cycles [0,1), `dt` is the phase increment per sample.
double poly_blep(double t, double dt) {
    if (t < dt) {
        t /= dt;
        return t + t - t * t - 1.0;
    }
    if (t > 1.0 - dt) {
        t = (t - 1.0) / dt;
        return t * t + t + t + 1.0;
    }
    return 0.0;
}

struct Oscillator {
    double phase = 0.0;  // cycles, [0,1)

    // Returns the next sample in [-1,1] and advances the phase.
    double next(int wave, double cyclesPerSample) {
        const double p = phase;
        double v;
        switch (wave) {
            case SYNTH_WAVE_SAW:
                v = 2.0 * p - 1.0 - poly_blep(p, cyclesPerSample);
                break;
            case SYNTH_WAVE_SQUARE: {
                v = p < 0.5 ? 1.0 : -1.0;
                v += poly_blep(p, cyclesPerSample);
                double q = p + 0.5;
                if (q >= 1.0) q -= 1.0;
                v -= poly_blep(q, cyclesPerSample);
                break;
            }
            case SYNTH_WAVE_TRIANGLE:
                v = 4.0 * std::fabs(p - 0.5) - 1.0;
                break;
            default:  // SYNTH_WAVE_SINE
                v = std::sin(kTwoPi * p);
                break;
        }
        phase += cyclesPerSample;
        if (phase >= 1.0) phase -= 1.0;
        return v;
    }
};

int clamp_int(float v, int lo, int hi) {
    int i = static_cast<int>(std::lround(v));
    if (i < lo) return lo;
    if (i > hi) return hi;
    return i;
}

// Topology-preserving state-variable filter (Zavalishin's TPT form). Stable at
// any cutoff below Nyquist, which matters because the cutoff can be swept to
// the top of its range. Only the low-pass output is used.
struct SvfLowpass {
    double ic1 = 0.0;  // integrator 1 state
    double ic2 = 0.0;  // integrator 2 state

    // g: pre-warped cutoff, k: damping (1/Q)
    double process(double x, double g, double k) {
        const double a1 = 1.0 / (1.0 + g * (g + k));
        const double a2 = g * a1;
        const double a3 = g * a2;
        const double v3 = x - ic2;
        const double v1 = a1 * ic1 + a2 * v3;
        const double v2 = ic2 + a2 * ic1 + a3 * v3;
        ic1 = 2.0 * v1 - ic1;
        ic2 = 2.0 * v2 - ic2;
        return v2;
    }
};

}  // namespace

float synth_cutoff_hz(float normalized) {
    if (normalized < 0.0f) normalized = 0.0f;
    if (normalized > 1.0f) normalized = 1.0f;
    return static_cast<float>(kMinCutoffHz * std::pow(kMaxCutoffHz / kMinCutoffHz, normalized));
}

float synth_envelope_seconds(float normalized) {
    if (normalized < 0.0f) normalized = 0.0f;
    if (normalized > 1.0f) normalized = 1.0f;
    return static_cast<float>(kMinEnvSeconds * std::pow(kMaxEnvSeconds / kMinEnvSeconds, normalized));
}

struct SynthCore {
    double sampleRate = 44100.0;
    double cyclesPerSample = 0.0;
    Oscillator oscA;
    Oscillator oscB;
    SvfLowpass filter;

    // Amplitude envelope (ADSR). Level runs 0-1 and is scaled by the note's velocity.
    int envStage = kEnvIdle;
    double envLevel = 0.0;
    double velocityGain = 0.0;

    int activeNote = -1;

    // Written by the UI thread, read by the render thread.
    std::atomic<float> params[SYNTH_PARAM_COUNT];
};

// ---- Parameter schema ------------------------------------------------------

int synth_param_count(void) {
    return SYNTH_PARAM_COUNT;
}

SynthParamInfo synth_param_info(int index) {
    if (index < 0 || index >= SYNTH_PARAM_COUNT) {
        SynthParamInfo empty = {"", "", SYNTH_PARAM_CONTINUOUS, 0, 0, 0};
        return empty;
    }
    return kParams[index];
}

int synth_waveform_count(void) {
    return SYNTH_WAVE_COUNT;
}

const char* synth_waveform_name(int index) {
    if (index < 0 || index >= SYNTH_WAVE_COUNT) return "";
    return kWaveNames[index];
}

void synth_set_param(SynthCore* core, int index, float value) {
    if (!core || index < 0 || index >= SYNTH_PARAM_COUNT) return;
    const SynthParamInfo& info = kParams[index];
    if (value < info.min) value = info.min;
    if (value > info.max) value = info.max;
    core->params[index].store(value, std::memory_order_relaxed);
}

float synth_get_param(SynthCore* core, int index) {
    if (!core || index < 0 || index >= SYNTH_PARAM_COUNT) return 0.0f;
    return core->params[index].load(std::memory_order_relaxed);
}

// ---- Lifecycle and performance --------------------------------------------

SynthCore* synth_create(double sample_rate) {
    SynthCore* core = new SynthCore();
    core->sampleRate = sample_rate;
    for (int i = 0; i < SYNTH_PARAM_COUNT; ++i) {
        core->params[i].store(kParams[i].defaultValue, std::memory_order_relaxed);
    }
    return core;
}

void synth_destroy(SynthCore* core) {
    delete core;
}

void synth_set_sample_rate(SynthCore* core, double sample_rate) {
    if (core) core->sampleRate = sample_rate;
}

void synth_note_on(SynthCore* core, int note, int velocity) {
    if (velocity <= 0) {
        synth_note_off(core, note);
        return;
    }
    const double hz = 440.0 * std::pow(2.0, (note - 69) / 12.0);
    core->activeNote = note;
    core->cyclesPerSample = hz / core->sampleRate;
    core->velocityGain = kMasterAmplitude * (velocity / 127.0);
    // Attack starts from the current level, so a retrigger does not click.
    core->envStage = kEnvAttack;
}

void synth_note_off(SynthCore* core, int note) {
    if (note != core->activeNote) {
        return;
    }
    core->activeNote = -1;
    if (core->envStage != kEnvIdle) {
        // Release starts from the current level, wherever the envelope is.
        core->envStage = kEnvRelease;
    }
}

void synth_render(SynthCore* core, float* out, int frames) {
    // Read parameters once per block.
    const int waveA = clamp_int(core->params[SYNTH_PARAM_OSC_A_WAVE].load(std::memory_order_relaxed),
                                0, SYNTH_WAVE_COUNT - 1);
    const int waveB = clamp_int(core->params[SYNTH_PARAM_OSC_B_WAVE].load(std::memory_order_relaxed),
                                0, SYNTH_WAVE_COUNT - 1);
    const double levelA = core->params[SYNTH_PARAM_OSC_A_LEVEL].load(std::memory_order_relaxed);
    const double levelB = core->params[SYNTH_PARAM_OSC_B_LEVEL].load(std::memory_order_relaxed);
    const double cps = core->cyclesPerSample;

    // Filter coefficients, once per block. Cutoff is clamped below Nyquist so
    // tan() stays finite and the response stays sensible.
    const double cutoffHz = synth_cutoff_hz(
        core->params[SYNTH_PARAM_FILTER_CUTOFF].load(std::memory_order_relaxed));
    const double resonance = core->params[SYNTH_PARAM_FILTER_RESONANCE].load(std::memory_order_relaxed);
    const double fc = std::fmin(cutoffHz, 0.45 * core->sampleRate);
    const double g = std::tan(3.141592653589793 * fc / core->sampleRate);
    const double k = kFlatDamping + (kMinDamping - kFlatDamping) * resonance;

    // Envelope rates, once per block. Each stage moves the level by a fixed
    // fraction of full scale per sample, so a longer time means a slower slope.
    const double sr = core->sampleRate;
    const double attackStep = 1.0 / (synth_envelope_seconds(
        core->params[SYNTH_PARAM_ENV_ATTACK].load(std::memory_order_relaxed)) * sr);
    const double decayStep = 1.0 / (synth_envelope_seconds(
        core->params[SYNTH_PARAM_ENV_DECAY].load(std::memory_order_relaxed)) * sr);
    const double releaseStep = 1.0 / (synth_envelope_seconds(
        core->params[SYNTH_PARAM_ENV_RELEASE].load(std::memory_order_relaxed)) * sr);
    const double sustain = core->params[SYNTH_PARAM_ENV_SUSTAIN].load(std::memory_order_relaxed);

    for (int i = 0; i < frames; ++i) {
        // ADSR envelope.
        switch (core->envStage) {
            case kEnvAttack:
                core->envLevel += attackStep;
                if (core->envLevel >= 1.0) {
                    core->envLevel = 1.0;
                    core->envStage = kEnvDecay;
                }
                break;
            case kEnvDecay:
                core->envLevel -= decayStep;
                if (core->envLevel <= sustain) {
                    core->envLevel = sustain;
                    core->envStage = kEnvSustain;
                }
                break;
            case kEnvSustain:
                core->envLevel = sustain;
                break;
            case kEnvRelease:
                core->envLevel -= releaseStep;
                if (core->envLevel <= 0.0) {
                    core->envLevel = 0.0;
                    core->envStage = kEnvIdle;
                }
                break;
            default:  // kEnvIdle
                core->envLevel = 0.0;
                break;
        }

        // Mixer: each oscillator scaled by its level, then summed.
        const double mixed = levelA * core->oscA.next(waveA, cps) +
                             levelB * core->oscB.next(waveB, cps);
        // One filter on the mixed signal, then the amplitude envelope.
        const double filtered = core->filter.process(mixed, g, k);
        out[i] = static_cast<float>(core->velocityGain * core->envLevel * filtered);
    }
}
