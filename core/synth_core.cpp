#include "synth_core.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <vector>

namespace {

const double kPi = 3.141592653589793;
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
    // Amp envelope. Defaults: attack 5 ms, decay 100 ms, sustain full, release 50 ms.
    {"env_attack",  "Attack",  SYNTH_PARAM_CONTINUOUS, 0.0f, 1.0f, 0.1748f},
    {"env_decay",   "Decay",   SYNTH_PARAM_CONTINUOUS, 0.0f, 1.0f, 0.5f},
    {"env_sustain", "Sustain", SYNTH_PARAM_CONTINUOUS, 0.0f, 1.0f, 1.0f},
    {"env_release", "Release", SYNTH_PARAM_CONTINUOUS, 0.0f, 1.0f, 0.4257f},
    {"oscA_detune", "Osc A Detune", SYNTH_PARAM_CONTINUOUS, -100.0f, 100.0f, 0.0f},
    {"oscB_detune", "Osc B Detune", SYNTH_PARAM_CONTINUOUS, -100.0f, 100.0f, 0.0f},
    // Filter envelope. Defaults give a closed filter until the amount is raised.
    {"filter_env_amount",  "Filter Env Amount",  SYNTH_PARAM_CONTINUOUS, 0.0f, 1.0f, 0.0f},
    {"filter_env_attack",  "Filter Attack",      SYNTH_PARAM_CONTINUOUS, 0.0f, 1.0f, 0.1748f},
    {"filter_env_decay",   "Filter Decay",       SYNTH_PARAM_CONTINUOUS, 0.0f, 1.0f, 0.5f},
    {"filter_env_sustain", "Filter Sustain",     SYNTH_PARAM_CONTINUOUS, 0.0f, 1.0f, 0.0f},
    {"filter_env_release", "Filter Release",     SYNTH_PARAM_CONTINUOUS, 0.0f, 1.0f, 0.4257f},
    // LFO. Defaults: 1 Hz, no modulation.
    {"lfo_rate",         "LFO Rate",          SYNTH_PARAM_CONTINUOUS, 0.0f, 1.0f, 0.5f},
    {"lfo_pitch_depth",  "LFO Pitch Depth",   SYNTH_PARAM_CONTINUOUS, 0.0f, 100.0f, 0.0f},
    {"lfo_filter_depth", "LFO Filter Depth",  SYNTH_PARAM_CONTINUOUS, 0.0f, 1.0f, 0.0f},
    // Effects. Defaults: clean, dry.
    {"drive",        "Drive",        SYNTH_PARAM_CONTINUOUS, 0.0f, 1.0f, 0.0f},
    {"reverb_mix",   "Reverb Mix",   SYNTH_PARAM_CONTINUOUS, 0.0f, 1.0f, 0.0f},
    {"reverb_size",  "Reverb Size",  SYNTH_PARAM_CONTINUOUS, 0.0f, 1.0f, 0.5f},
    // Neural source. Defaults to 0 so the oscillators alone make the sound.
    {"source_mix",   "Source Mix",   SYNTH_PARAM_CONTINUOUS, 0.0f, 1.0f, 0.0f},
    // Movement of the neural source. Defaults to full, so a timbre with movement plays it.
    {"movement",     "Movement",     SYNTH_PARAM_CONTINUOUS, 0.0f, 1.0f, 1.0f},
};

const double kMinCutoffHz = 20.0;
const double kMaxCutoffHz = 20000.0;
const double kMinEnvSeconds = 0.001;
const double kMaxEnvSeconds = 10.0;
const double kMinLfoHz = 0.05;
const double kMaxLfoHz = 20.0;

// Resonance maps to the SVF damping term k (k = 1/Q). k = sqrt(2) is a flat,
// Butterworth response. Lower k gives a peak at the cutoff. k is kept at
// 0.1 or above (Q up to 10) so the filter stays well behaved.
const double kFlatDamping = 1.4142135623730951;
const double kMinDamping = 0.1;

// Filter envelope and LFO range, in octaves of cutoff.
const double kFilterEnvOctaves = 5.0;
const double kLfoFilterOctaves = 2.0;

// Drive: soft clip with input gain from 1x (clean) to 10x.
const double kDriveMaxGain = 9.0;

// Reverb (Freeverb-style): 8 damped combs and 4 allpasses, mono.
// Tunings are for 44.1 kHz and are scaled to the actual sample rate.
const int kCombTunings[8] = {1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617};
const int kAllpassTunings[4] = {556, 441, 341, 225};
const double kTuningRate = 44100.0;
const double kReverbDamp = 0.4;
const double kReverbInputGain = 0.015;
const double kReverbWetScale = 3.0;
const double kReverbFeedbackMin = 0.7;
const double kReverbFeedbackRange = 0.28;

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

// ADSR envelope. Level runs 0-1. Attack and release start from the current
// level, so retriggering or releasing early does not click.
struct Envelope {
    int stage = kEnvIdle;
    double level = 0.0;

    void noteOn() { stage = kEnvAttack; }

    void noteOff() {
        if (stage != kEnvIdle) stage = kEnvRelease;
    }

    // Steps are the change in level per sample for each stage.
    void tick(double attackStep, double decayStep, double releaseStep, double sustain) {
        switch (stage) {
            case kEnvAttack:
                level += attackStep;
                if (level >= 1.0) {
                    level = 1.0;
                    stage = kEnvDecay;
                }
                break;
            case kEnvDecay:
                level -= decayStep;
                if (level <= sustain) {
                    level = sustain;
                    stage = kEnvSustain;
                }
                break;
            case kEnvSustain:
                level = sustain;
                break;
            case kEnvRelease:
                level -= releaseStep;
                if (level <= 0.0) {
                    level = 0.0;
                    stage = kEnvIdle;
                }
                break;
            default:  // kEnvIdle
                level = 0.0;
                break;
        }
    }
};

// Topology-preserving state-variable filter (Zavalishin's TPT form). Stable at
// any cutoff below Nyquist. Only the low-pass output is used.
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

struct CombFilter {
    std::vector<double> buf;
    int idx = 0;
    double store = 0.0;

    void resize(int n) {
        buf.assign(static_cast<size_t>(n), 0.0);
        idx = 0;
        store = 0.0;
    }

    void clear() {
        std::fill(buf.begin(), buf.end(), 0.0);
        idx = 0;
        store = 0.0;
    }

    double process(double input, double feedback, double damp) {
        const double out = buf[idx];
        store = out * (1.0 - damp) + store * damp;
        buf[idx] = input + store * feedback;
        if (++idx >= static_cast<int>(buf.size())) idx = 0;
        return out;
    }
};

struct AllpassFilter {
    std::vector<double> buf;
    int idx = 0;

    void resize(int n) {
        buf.assign(static_cast<size_t>(n), 0.0);
        idx = 0;
    }

    void clear() {
        std::fill(buf.begin(), buf.end(), 0.0);
        idx = 0;
    }

    double process(double input) {
        const double bufout = buf[idx];
        const double out = -input + bufout;
        buf[idx] = input + bufout * 0.5;
        if (++idx >= static_cast<int>(buf.size())) idx = 0;
        return out;
    }
};

// Mono Freeverb-style reverb. Buffers are allocated in configure(), which is
// called off the audio thread, so process() never allocates.
struct Reverb {
    CombFilter combs[8];
    AllpassFilter allpasses[4];

    void configure(double sampleRate) {
        const double scale = sampleRate / kTuningRate;
        for (int i = 0; i < 8; ++i) {
            combs[i].resize(static_cast<int>(kCombTunings[i] * scale));
        }
        for (int i = 0; i < 4; ++i) {
            allpasses[i].resize(static_cast<int>(kAllpassTunings[i] * scale));
        }
    }

    void clear() {
        for (auto& c : combs) c.clear();
        for (auto& a : allpasses) a.clear();
    }

    // Returns the wet signal only. The caller mixes it with the dry signal.
    double processWet(double input, double feedback) {
        const double x = input * kReverbInputGain;
        double sum = 0.0;
        for (auto& c : combs) sum += c.process(x, feedback, kReverbDamp);
        for (auto& a : allpasses) sum = a.process(sum);
        return sum * kReverbWetScale;
    }
};

// ---- Neural source (timbre) wavetables -------------------------------------
// The timbre is a set of harmonic amplitudes. It is turned into one bandlimited
// single-cycle wavetable per octave band, so a high note never contains
// harmonics above Nyquist. Bands are built off the audio thread and swapped in
// with an atomic index (see SynthCore::timbreFront).

const int kWaveTableSize = 2048;
const int kWaveBands = 10;
const double kBandBaseHz = 40.0;   // band b covers fundamentals up to 40 Hz * 2^b
const double kTimbreNyquistLimit = 0.45;  // fraction of the sample rate

struct TimbreSet {
    float harmonics[SYNTH_TIMBRE_MAX_HARMONICS] = {};
    // One extra sample per table so linear interpolation can wrap without a branch.
    float tables[kWaveBands][kWaveTableSize + 1] = {};
    // Loudness over the note (see synth_set_timbre_shaped). envCount 0 = no envelope.
    float envelope[SYNTH_TIMBRE_MAX_ENV_POINTS] = {};
    int envCount = 0;
    // Movement tracks (see synth_set_timbre_moving). moveFrames 0 = no movement.
    float moveAmp[SYNTH_MOVEMENT_MAX_FRAMES] = {};
    float movePitch[SYNTH_MOVEMENT_MAX_FRAMES] = {};
    float moveBright[SYNTH_MOVEMENT_MAX_FRAMES] = {};
    int moveFrames = 0;
    int moveLoopStart = 0;
    int moveLoopEnd = 0;
};

// Copies an envelope into `set`. A null or empty envelope clears it.
void set_envelope(TimbreSet& set, const float* envelope, int count) {
    if (!envelope || count <= 0) {
        set.envCount = 0;
        return;
    }
    if (count > SYNTH_TIMBRE_MAX_ENV_POINTS) count = SYNTH_TIMBRE_MAX_ENV_POINTS;
    for (int k = 0; k < count; ++k) set.envelope[k] = envelope[k];
    set.envCount = count;
}

// Copies movement tracks into `set`. A null movement, or one with missing tracks
// or no frames, clears it. An invalid loop region falls back to the whole clip.
void set_movement(TimbreSet& set, const SynthMovement* m) {
    if (!m || m->frameCount <= 0 || !m->amp || !m->pitchCents || !m->brightnessOct) {
        set.moveFrames = 0;
        return;
    }
    const int n = m->frameCount > SYNTH_MOVEMENT_MAX_FRAMES ? SYNTH_MOVEMENT_MAX_FRAMES : m->frameCount;
    for (int k = 0; k < n; ++k) {
        set.moveAmp[k] = m->amp[k];
        set.movePitch[k] = m->pitchCents[k];
        set.moveBright[k] = m->brightnessOct[k];
    }
    set.moveFrames = n;
    int ls = m->loopStart, le = m->loopEnd;
    if (ls < 0 || le > n || le <= ls) {
        ls = 0;
        le = n;
    }
    set.moveLoopStart = ls;
    set.moveLoopEnd = le;
}

// Copies the movement tracks of `src` into `dst`.
void copy_movement(TimbreSet& dst, const TimbreSet& src) {
    const int n = src.moveFrames;
    if (n <= 0) {
        dst.moveFrames = 0;
        return;
    }
    for (int k = 0; k < n; ++k) {
        dst.moveAmp[k] = src.moveAmp[k];
        dst.movePitch[k] = src.movePitch[k];
        dst.moveBright[k] = src.moveBright[k];
    }
    dst.moveFrames = n;
    dst.moveLoopStart = src.moveLoopStart;
    dst.moveLoopEnd = src.moveLoopEnd;
}

// The three movement values at one frame position (linearly interpolated).
struct MovementPoint {
    double amp;
    double pitchCents;
    double brightnessOct;
};

// Movement at frame position `frame` (0 = note-on). Frames before the loop play
// once; from loopStart on, positions wrap inside [loopStart, loopEnd).
MovementPoint movement_point(const TimbreSet& set, double frame) {
    const int ls = set.moveLoopStart;
    const int le = set.moveLoopEnd;
    double p = frame;
    if (p >= le) p = ls + std::fmod(p - ls, static_cast<double>(le - ls));
    int i0 = static_cast<int>(p);
    if (i0 >= le) i0 = le - 1;
    if (i0 < 0) i0 = 0;
    const double f = p - i0;
    int i1 = i0 + 1;
    if (i1 >= le) i1 = ls;
    auto lerp = [f, i0, i1](const float* t) { return t[i0] + f * (t[i1] - t[i0]); };
    return {lerp(set.moveAmp), lerp(set.movePitch), lerp(set.moveBright)};
}

// Level of the envelope at `t`, where t = 0 is note-on and t = 1 is the end of
// SYNTH_TIMBRE_ENV_SECONDS. Linear between points, held at the last point after the end.
double envelope_at(const TimbreSet& set, double t) {
    const int last = set.envCount - 1;
    if (t <= 0.0) return set.envelope[0];
    const double pos = t * last;
    if (pos >= last) return set.envelope[last];
    const int i = static_cast<int>(pos);
    const double frac = pos - i;
    return set.envelope[i] + frac * (set.envelope[i + 1] - set.envelope[i]);
}

// Builds all bands from `harmonics` into `set`, at sample rate `sr`.
// Allocates nothing: the storage lives inside TimbreSet.
void build_timbre(TimbreSet& set, const float* harmonics, int count, double sr) {
    for (int k = 0; k < SYNTH_TIMBRE_MAX_HARMONICS; ++k) {
        set.harmonics[k] = (k < count) ? harmonics[k] : 0.0f;
    }

    // Normalize so the full (all-harmonic) cycle peaks at 1.0. Every band uses
    // the same scale, so a band switch does not change the loudness.
    double fullPeak = 0.0;
    for (int n = 0; n < kWaveTableSize; ++n) {
        double s = 0.0;
        for (int k = 1; k <= SYNTH_TIMBRE_MAX_HARMONICS; ++k) {
            s += set.harmonics[k - 1] * std::sin(kTwoPi * k * n / kWaveTableSize);
        }
        fullPeak = std::fmax(fullPeak, std::fabs(s));
    }
    const double scale = fullPeak > 0.0 ? 1.0 / fullPeak : 0.0;

    for (int b = 0; b < kWaveBands; ++b) {
        const double topHz = kBandBaseHz * std::pow(2.0, b);
        int kmax = static_cast<int>(std::floor(kTimbreNyquistLimit * sr / topHz));
        if (kmax > SYNTH_TIMBRE_MAX_HARMONICS) kmax = SYNTH_TIMBRE_MAX_HARMONICS;
        for (int n = 0; n < kWaveTableSize; ++n) {
            double s = 0.0;
            for (int k = 1; k <= kmax; ++k) {
                s += set.harmonics[k - 1] * std::sin(kTwoPi * k * n / kWaveTableSize);
            }
            set.tables[b][n] = static_cast<float>(s * scale);
        }
        set.tables[b][kWaveTableSize] = set.tables[b][0];
    }
}

// Band whose table is alias-free for a note at `hz`.
int band_for_hz(double hz) {
    for (int b = 0; b < kWaveBands - 1; ++b) {
        if (hz <= kBandBaseHz * std::pow(2.0, b)) return b;
    }
    return kWaveBands - 1;
}

// Reads the wavetable for a note at `hz` (`cps` cycles per sample) and advances `phase`.
double read_timbre(const TimbreSet& set, double hz, double cps, double& phase) {
    const int band = band_for_hz(hz);
    const double pos = phase * kWaveTableSize;
    int i = static_cast<int>(pos);
    const double frac = pos - i;
    const float* t = set.tables[band];
    const double out = t[i] + frac * (t[i + 1] - t[i]);
    phase += cps;
    if (phase >= 1.0) phase -= 1.0;
    return out;
}

// One sounding note: its own oscillators, filter and envelopes. Voices are
// summed, then drive and reverb are applied to the mix.
struct Voice {
    int note = -1;
    double baseHz = 0.0;
    double velocityGain = 0.0;
    bool held = false;               // key is still down (not yet released)
    unsigned long long order = 0;    // start order, used to steal the oldest voice
    Oscillator oscA;
    Oscillator oscB;
    double tablePhase = 0.0;         // neural source phase, cycles [0,1)
    double envSamples = 0.0;         // samples since note-on, for the timbre envelope
    SvfLowpass filter;
    Envelope ampEnv;
    Envelope filterEnv;

    bool sounding() const { return ampEnv.stage != kEnvIdle; }
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

float synth_lfo_hz(float normalized) {
    if (normalized < 0.0f) normalized = 0.0f;
    if (normalized > 1.0f) normalized = 1.0f;
    return static_cast<float>(kMinLfoHz * std::pow(kMaxLfoHz / kMinLfoHz, normalized));
}

struct SynthCore {
    static const int kMaxVoices = 8;

    double sampleRate = 44100.0;
    Voice voices[kMaxVoices];
    unsigned long long nextOrder = 1;
    double lfoPhase = 0.0;     // cycles, [0,1)
    Reverb reverb;

    // Two timbre sets. The render thread reads timbreFront once per block; the
    // UI thread builds into the other set and then publishes it. -1 = no timbre.
    // Known limit: two publishes within one block could overlap the set the
    // render thread is still reading. Fine for UI-driven loads; step 4 needs a
    // proper triple buffer before timbres change during exploration.
    TimbreSet timbreSets[2];
    std::atomic<int> timbreFront{-1};

    // Written by the UI thread, read by the render thread.
    std::atomic<float> params[SYNTH_PARAM_COUNT];
};

// Converts a normalized envelope parameter to a per-sample step.
static double env_step(float normalized, double sampleRate) {
    return 1.0 / (synth_envelope_seconds(normalized) * sampleRate);
}

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
    core->reverb.configure(sample_rate);
    for (int i = 0; i < SYNTH_PARAM_COUNT; ++i) {
        core->params[i].store(kParams[i].defaultValue, std::memory_order_relaxed);
    }
    return core;
}

void synth_destroy(SynthCore* core) {
    delete core;
}

void synth_reset(SynthCore* core) {
    if (!core) return;
    for (auto& v : core->voices) v = Voice();
    core->lfoPhase = 0.0;
    core->reverb.clear();
}

void synth_set_sample_rate(SynthCore* core, double sample_rate) {
    if (!core) return;
    core->sampleRate = sample_rate;
    core->reverb.configure(sample_rate);  // off the audio thread
    // The bandlimit depends on the sample rate, so rebuild a loaded timbre.
    const int front = core->timbreFront.load(std::memory_order_acquire);
    if (front >= 0) {
        const int back = 1 - front;
        const TimbreSet& old = core->timbreSets[front];
        build_timbre(core->timbreSets[back], old.harmonics, SYNTH_TIMBRE_MAX_HARMONICS, sample_rate);
        set_envelope(core->timbreSets[back], old.envelope, old.envCount);
        copy_movement(core->timbreSets[back], old);
        core->timbreFront.store(back, std::memory_order_release);
    }
}

// Builds a timbre into the back set and publishes it. Shared by the setters below.
void publish_timbre(SynthCore* core, const float* harmonics, int count,
                    const float* envelope, int envCount, const SynthMovement* movement) {
    if (count <= 0 || !harmonics) {
        core->timbreFront.store(-1, std::memory_order_release);
        return;
    }
    if (count > SYNTH_TIMBRE_MAX_HARMONICS) count = SYNTH_TIMBRE_MAX_HARMONICS;
    const int front = core->timbreFront.load(std::memory_order_acquire);
    const int back = front == 0 ? 1 : 0;
    build_timbre(core->timbreSets[back], harmonics, count, core->sampleRate);
    set_envelope(core->timbreSets[back], envelope, envCount);
    set_movement(core->timbreSets[back], movement);
    core->timbreFront.store(back, std::memory_order_release);
}

void synth_set_timbre_shaped(SynthCore* core, const float* harmonics, int count,
                             const float* envelope, int envCount) {
    if (!core) return;
    publish_timbre(core, harmonics, count, envelope, envCount, nullptr);
}

void synth_set_timbre_moving(SynthCore* core, const float* harmonics, int count,
                             const SynthMovement* movement) {
    if (!core) return;
    publish_timbre(core, harmonics, count, nullptr, 0, movement);
}

void synth_set_timbre(SynthCore* core, const float* harmonics, int count) {
    synth_set_timbre_shaped(core, harmonics, count, nullptr, 0);
}

void synth_note_on(SynthCore* core, int note, int velocity) {
    if (velocity <= 0) {
        synth_note_off(core, note);
        return;
    }
    // Pick a voice: an idle one first, then the oldest released one, then the
    // oldest one overall (stealing).
    Voice* chosen = nullptr;
    for (auto& v : core->voices) {
        if (!v.sounding()) { chosen = &v; break; }
    }
    if (!chosen) {
        for (auto& v : core->voices) {
            if (!v.held && (!chosen || v.order < chosen->order)) chosen = &v;
        }
    }
    if (!chosen) {
        for (auto& v : core->voices) {
            if (!chosen || v.order < chosen->order) chosen = &v;
        }
    }

    chosen->note = note;
    chosen->held = true;
    chosen->order = core->nextOrder++;
    chosen->baseHz = 440.0 * std::pow(2.0, (note - 69) / 12.0);
    // Start each note at the same oscillator phase and filter state, so a patch
    // always sounds the same when played the same way (needed for deterministic renders).
    chosen->oscA.phase = 0.0;
    chosen->oscB.phase = 0.0;
    chosen->tablePhase = 0.0;
    chosen->envSamples = 0.0;
    chosen->filter = SvfLowpass();
    chosen->velocityGain = kMasterAmplitude * (velocity / 127.0);
    chosen->ampEnv.noteOn();
    chosen->filterEnv.noteOn();
}

void synth_note_off(SynthCore* core, int note) {
    // Release the oldest held voice playing this note.
    Voice* target = nullptr;
    for (auto& v : core->voices) {
        if (v.held && v.note == note && (!target || v.order < target->order)) target = &v;
    }
    if (!target) return;
    target->held = false;
    target->ampEnv.noteOff();
    target->filterEnv.noteOff();
}

void synth_render(SynthCore* core, float* out, int frames) {
    const double sr = core->sampleRate;
    auto load = [core](int i) {
        return static_cast<double>(core->params[i].load(std::memory_order_relaxed));
    };

    // Read parameters once per block.
    const int waveA = clamp_int(static_cast<float>(load(SYNTH_PARAM_OSC_A_WAVE)), 0, SYNTH_WAVE_COUNT - 1);
    const int waveB = clamp_int(static_cast<float>(load(SYNTH_PARAM_OSC_B_WAVE)), 0, SYNTH_WAVE_COUNT - 1);
    const double levelA = load(SYNTH_PARAM_OSC_A_LEVEL);
    const double levelB = load(SYNTH_PARAM_OSC_B_LEVEL);
    const double detuneA = load(SYNTH_PARAM_OSC_A_DETUNE);
    const double detuneB = load(SYNTH_PARAM_OSC_B_DETUNE);

    const double baseCutoffHz = synth_cutoff_hz(static_cast<float>(load(SYNTH_PARAM_FILTER_CUTOFF)));
    const double resonance = load(SYNTH_PARAM_FILTER_RESONANCE);
    const double k = kFlatDamping + (kMinDamping - kFlatDamping) * resonance;
    const double filterEnvOctaves = load(SYNTH_PARAM_FILTER_ENV_AMOUNT) * kFilterEnvOctaves;

    const double lfoInc = synth_lfo_hz(static_cast<float>(load(SYNTH_PARAM_LFO_RATE))) / sr;
    const double lfoPitchCents = load(SYNTH_PARAM_LFO_PITCH_DEPTH);
    const double lfoFilterOctaves = load(SYNTH_PARAM_LFO_FILTER_DEPTH) * kLfoFilterOctaves;

    const double drive = load(SYNTH_PARAM_DRIVE);
    const double driveGain = 1.0 + kDriveMaxGain * drive;
    const double driveNorm = std::tanh(driveGain);

    const double reverbMix = load(SYNTH_PARAM_REVERB_MIX);
    const double reverbFeedback = kReverbFeedbackMin + kReverbFeedbackRange * load(SYNTH_PARAM_REVERB_SIZE);

    // Neural source: how much of the timbre to hear, and which table set to read.
    const double sourceMix = std::fmin(std::fmax(load(SYNTH_PARAM_SOURCE_MIX), 0.0), 1.0);
    const int timbreIndex = core->timbreFront.load(std::memory_order_acquire);
    const TimbreSet* timbre = timbreIndex >= 0 ? &core->timbreSets[timbreIndex] : nullptr;
    const double movement = std::fmin(std::fmax(load(SYNTH_PARAM_MOVEMENT), 0.0), 1.0);

    // Envelope steps, once per block.
    const double ampAttack = env_step(static_cast<float>(load(SYNTH_PARAM_ENV_ATTACK)), sr);
    const double ampDecay = env_step(static_cast<float>(load(SYNTH_PARAM_ENV_DECAY)), sr);
    const double ampRelease = env_step(static_cast<float>(load(SYNTH_PARAM_ENV_RELEASE)), sr);
    const double ampSustain = load(SYNTH_PARAM_ENV_SUSTAIN);
    const double fltAttack = env_step(static_cast<float>(load(SYNTH_PARAM_FILTER_ENV_ATTACK)), sr);
    const double fltDecay = env_step(static_cast<float>(load(SYNTH_PARAM_FILTER_ENV_DECAY)), sr);
    const double fltRelease = env_step(static_cast<float>(load(SYNTH_PARAM_FILTER_ENV_RELEASE)), sr);
    const double fltSustain = load(SYNTH_PARAM_FILTER_ENV_SUSTAIN);

    for (int i = 0; i < frames; ++i) {
        // LFO: sine, free running. Shared by all voices.
        const double lfo = std::sin(kTwoPi * core->lfoPhase);
        core->lfoPhase += lfoInc;
        if (core->lfoPhase >= 1.0) core->lfoPhase -= 1.0;

        // Sum the sounding voices. Each one: oscillators, then its own filter, then its amp.
        double mixed = 0.0;
        for (auto& v : core->voices) {
            if (!v.sounding()) continue;
            v.ampEnv.tick(ampAttack, ampDecay, ampRelease, ampSustain);
            v.filterEnv.tick(fltAttack, fltDecay, fltRelease, fltSustain);

            const double baseCps = v.baseHz / sr;
            // Pitch: detune plus LFO vibrato. Cents -> frequency ratio is 2^(c/1200).
            const double cpsA = baseCps * std::exp2((detuneA + lfoPitchCents * lfo) / 1200.0);
            const double cpsB = baseCps * std::exp2((detuneB + lfoPitchCents * lfo) / 1200.0);
            const double oscMix = levelA * v.oscA.next(waveA, cpsA) +
                                  levelB * v.oscB.next(waveB, cpsB);
            // Neural source follows osc A's pitch, including detune and vibrato.
            double raw = (1.0 - sourceMix) * oscMix;
            double brightOct = 0.0;  // movement brightness offset for this voice's filter
            if (sourceMix > 0.0 && timbre) {
                double gain = 1.0;
                double pitchCents = 0.0;
                if (timbre->moveFrames > 0) {
                    // Movement replaces the envelope. Frame 0 is note-on, at SYNTH_MOVEMENT_RATE_HZ.
                    const MovementPoint at = movement_point(
                        *timbre, v.envSamples * SYNTH_MOVEMENT_RATE_HZ / sr);
                    gain = 1.0 + movement * (at.amp - 1.0);
                    pitchCents = movement * at.pitchCents;
                    brightOct = sourceMix * movement * at.brightnessOct;
                } else if (timbre->envCount > 0) {
                    gain = envelope_at(*timbre, v.envSamples / (SYNTH_TIMBRE_ENV_SECONDS * sr));
                }
                const double cpsN = cpsA * std::exp2(pitchCents / 1200.0);
                raw += sourceMix * gain * read_timbre(*timbre, cpsN * sr, cpsN, v.tablePhase);
            }
            v.envSamples += 1.0;

            // Filter: base cutoff, opened by this voice's envelope, moved by the LFO
            // and by the source's brightness movement.
            double cutoff = baseCutoffHz *
                            std::exp2(filterEnvOctaves * v.filterEnv.level + lfoFilterOctaves * lfo + brightOct);
            cutoff = std::fmin(std::fmax(cutoff, kMinCutoffHz), 0.45 * sr);
            const double g = std::tan(kPi * cutoff / sr);
            const double filtered = v.filter.process(raw, g, k);

            mixed += v.velocityGain * v.ampEnv.level * filtered;
        }

        // Drive: soft clip on the mix, normalized so the peak stays near full scale.
        const double shaped = drive > 0.0 ? std::tanh(mixed * driveGain) / driveNorm : mixed;

        // Reverb after the mix so the tail rings out after the notes end.
        const double wet = core->reverb.processWet(shaped, reverbFeedback);
        out[i] = static_cast<float>(shaped * (1.0 - reverbMix) + wet * reverbMix);
    }
}
