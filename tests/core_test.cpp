// Core-level tests: parameters and MIDI-style calls in, samples out. No Apple APIs.
#include "../core/synth_core.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static int failures = 0;

#define CHECK(cond, msg)                                   \
    do {                                                   \
        if (cond) {                                        \
            std::printf("  ok   %s\n", msg);               \
        } else {                                           \
            std::printf("  FAIL %s\n", msg);               \
            ++failures;                                    \
        }                                                  \
    } while (0)

static const double kSR = 48000.0;
static const int kSecond = 48000;

// Estimates frequency from rising zero crossings.
static double estimate_hz(const std::vector<float>& x, double sr) {
    int crossings = 0;
    for (size_t i = 1; i < x.size(); ++i) {
        if (x[i - 1] <= 0.0f && x[i] > 0.0f) ++crossings;
    }
    return crossings * sr / static_cast<double>(x.size());
}

static double peak(const std::vector<float>& x) {
    double p = 0.0;
    for (float s : x) p = std::fmax(p, std::fabs(s));
    return p;
}

static double rms(const std::vector<float>& x) {
    double acc = 0.0;
    for (float s : x) acc += static_cast<double>(s) * s;
    return std::sqrt(acc / x.size());
}

// Fraction of samples sitting near the top or bottom of the waveform.
// Square is close to 1.0; sine, saw and triangle are much lower.
static double plateau_fraction(const std::vector<float>& x, double threshold) {
    size_t n = 0;
    for (float s : x) {
        if (std::fabs(s) > threshold) ++n;
    }
    return static_cast<double>(n) / x.size();
}

// Plays A4 with the given levels/waves and returns `frames` samples.
static std::vector<float> play_a4(int waveA, float levelA, int waveB, float levelB,
                                  int frames = kSecond) {
    SynthCore* s = synth_create(kSR);
    synth_set_param(s, SYNTH_PARAM_OSC_A_WAVE, static_cast<float>(waveA));
    synth_set_param(s, SYNTH_PARAM_OSC_A_LEVEL, levelA);
    synth_set_param(s, SYNTH_PARAM_OSC_B_WAVE, static_cast<float>(waveB));
    synth_set_param(s, SYNTH_PARAM_OSC_B_LEVEL, levelB);
    synth_note_on(s, 69, 127);  // full velocity: amplitude 0.5
    std::vector<float> out(frames);
    synth_render(s, out.data(), frames);
    synth_destroy(s);
    return out;
}

// Plays one note through the default oscillator with the given filter settings.
static std::vector<float> play_filtered(int note, float cutoff, float resonance,
                                        int wave = SYNTH_WAVE_SINE, int frames = kSecond) {
    SynthCore* s = synth_create(kSR);
    synth_set_param(s, SYNTH_PARAM_OSC_A_WAVE, static_cast<float>(wave));
    synth_set_param(s, SYNTH_PARAM_FILTER_CUTOFF, cutoff);
    synth_set_param(s, SYNTH_PARAM_FILTER_RESONANCE, resonance);
    synth_note_on(s, note, 127);
    std::vector<float> out(frames);
    synth_render(s, out.data(), frames);
    synth_destroy(s);
    return out;
}

// Plays one note with the given envelope. Holds the key for `holdFrames`, then
// releases it and renders `tailFrames` more. Envelope values are normalized.
static std::vector<float> render_env(int note, float attack, float decay, float sustain,
                                     float release, int holdFrames, int tailFrames) {
    SynthCore* s = synth_create(kSR);
    synth_set_param(s, SYNTH_PARAM_ENV_ATTACK, attack);
    synth_set_param(s, SYNTH_PARAM_ENV_DECAY, decay);
    synth_set_param(s, SYNTH_PARAM_ENV_SUSTAIN, sustain);
    synth_set_param(s, SYNTH_PARAM_ENV_RELEASE, release);
    synth_note_on(s, note, 127);
    std::vector<float> out(holdFrames + tailFrames);
    synth_render(s, out.data(), holdFrames);
    synth_note_off(s, note);
    synth_render(s, out.data() + holdFrames, tailFrames);
    synth_destroy(s);
    return out;
}

// Normalized cutoff that puts the filter at `hz`. Inverse of synth_cutoff_hz.
static float cutoff_for_hz(double hz) {
    return static_cast<float>(std::log(hz / 20.0) / std::log(1000.0));
}

int main() {
    std::printf("Filter schema and cutoff mapping\n");
    {
        CHECK(synth_param_count() == 25, "25 parameters: osc, filter, envelopes, LFO, effects, source mix and movement");
        CHECK(std::strcmp(synth_param_info(SYNTH_PARAM_REVERB_MIX).id, "reverb_mix") == 0,
              "reverb mix id");
        CHECK(synth_param_info(SYNTH_PARAM_OSC_A_DETUNE).min == -100.0f, "detune range is +/-100 cents");
        CHECK(synth_param_info(SYNTH_PARAM_DRIVE).defaultValue == 0.0f, "drive defaults to clean");
        CHECK(synth_param_info(SYNTH_PARAM_REVERB_MIX).defaultValue == 0.0f, "reverb defaults to dry");
        CHECK(std::strcmp(synth_param_info(SYNTH_PARAM_FILTER_CUTOFF).id, "filter_cutoff") == 0,
              "cutoff parameter id");
        CHECK(synth_param_info(SYNTH_PARAM_FILTER_CUTOFF).defaultValue == 1.0f,
              "cutoff defaults to fully open");
        CHECK(synth_param_info(SYNTH_PARAM_FILTER_RESONANCE).defaultValue == 0.0f,
              "resonance defaults to none");
        CHECK(std::fabs(synth_cutoff_hz(0.0f) - 20.0f) < 0.01f, "cutoff 0 is 20 Hz");
        CHECK(std::fabs(synth_cutoff_hz(1.0f) - 20000.0f) < 0.1f, "cutoff 1 is 20 kHz");
        CHECK(std::fabs(synth_cutoff_hz(0.5f) - 632.46f) < 0.1f, "cutoff 0.5 is the geometric mid, 632 Hz");
    }

    std::printf("Low-pass removes high frequencies\n");
    {
        // Saw at C7 (2093 Hz) is rich in harmonics. A 80 Hz cutoff removes
        // nearly all of them, while the fully open filter keeps them.
        double open = rms(play_filtered(96, 1.0f, 0.0f, SYNTH_WAVE_SAW));
        double closed = rms(play_filtered(96, cutoff_for_hz(80.0), 0.0f, SYNTH_WAVE_SAW));
        CHECK(closed < open * 0.05, "closed cutoff strongly attenuates a high saw");
    }

    std::printf("Low-pass keeps low frequencies\n");
    {
        // A4 sine sits well below a 5 kHz cutoff and should pass almost unchanged.
        double open = rms(play_filtered(69, 1.0f, 0.0f));
        double bright = rms(play_filtered(69, cutoff_for_hz(5000.0), 0.0f));
        CHECK(std::fabs(bright - open) / open < 0.01, "5 kHz cutoff leaves A4 untouched");
    }

    std::printf("Resonance boosts the cutoff region\n");
    {
        // A4 is 440 Hz. With the cutoff on it, resonance should raise the
        // level well above the flat (no resonance) response.
        float cutoff = cutoff_for_hz(440.0);
        double flat = rms(play_filtered(69, cutoff, 0.0f));
        double peaked = rms(play_filtered(69, cutoff, 1.0f));
        CHECK(peaked > flat * 3.0, "full resonance gives a strong peak at the cutoff");
    }

    std::printf("Filter stays stable at extremes\n");
    {
        // Highest cutoff, full resonance, bright saw: output must stay finite and bounded.
        std::vector<float> out = play_filtered(120, 1.0f, 1.0f, SYNTH_WAVE_SAW);
        bool finite = true;
        for (float s : out) finite = finite && std::isfinite(s);
        CHECK(finite, "output is finite at full cutoff and resonance");
        CHECK(peak(out) < 10.0, "output stays bounded at full resonance");
    }

    std::printf("Envelope times and schema\n");
    {
        CHECK(std::fabs(synth_envelope_seconds(0.0f) - 0.001f) < 1e-6f, "envelope 0 is 1 ms");
        CHECK(std::fabs(synth_envelope_seconds(1.0f) - 10.0f) < 1e-4f, "envelope 1 is 10 s");
        CHECK(std::fabs(synth_envelope_seconds(synth_param_info(SYNTH_PARAM_ENV_ATTACK).defaultValue) - 0.005f) < 1e-4f,
              "attack defaults to about 5 ms");
        CHECK(synth_param_info(SYNTH_PARAM_ENV_SUSTAIN).defaultValue == 1.0f, "sustain defaults to full");
        CHECK(std::strcmp(synth_param_info(SYNTH_PARAM_ENV_RELEASE).id, "env_release") == 0,
              "release parameter id");
    }

    std::printf("Attack: the note fades in\n");
    {
        // Attack about 500 ms. The first 10 ms should be much quieter than the held part.
        std::vector<float> out = render_env(69, 0.675f, 0.0f, 1.0f, 0.4257f, kSecond, 0);
        std::vector<float> early(out.begin(), out.begin() + 480);
        std::vector<float> late(out.begin() + 24000, out.begin() + 28800);
        CHECK(rms(early) < rms(late) * 0.1, "first 10 ms is far quieter than the held note");
    }

    std::printf("Decay and sustain set the held level\n");
    {
        // Fast decay to half sustain: after 0.5 s the note holds at 0.5 of full.
        std::vector<float> full = render_env(69, 0.1748f, 0.0f, 1.0f, 0.4257f, kSecond, 0);
        std::vector<float> half = render_env(69, 0.1748f, 0.0f, 0.5f, 0.4257f, kSecond, 0);
        std::vector<float> fullLate(full.begin() + 24000, full.end());
        std::vector<float> halfLate(half.begin() + 24000, half.end());
        CHECK(std::fabs(rms(halfLate) - rms(fullLate) * 0.5) < 0.01, "sustain 0.5 holds at half level");
    }

    std::printf("Decay to zero sustain goes silent while the key is held\n");
    {
        std::vector<float> out = render_env(69, 0.1748f, 0.0f, 0.0f, 0.4257f, kSecond, 0);
        std::vector<float> tail(out.begin() + 9600, out.begin() + 19200);  // 0.2 s - 0.4 s
        CHECK(peak(tail) == 0.0, "silent after decay with zero sustain");
    }

    std::printf("Release: the tail fades out after note off\n");
    {
        // Release about 500 ms. Hold 0.2 s, release, then check the fade and the end.
        std::vector<float> out = render_env(69, 0.1748f, 0.0f, 1.0f, 0.675f, 9600, 36000);
        std::vector<float> mid(out.begin() + 9600 + 4800, out.begin() + 9600 + 7200);   // 100 ms after off
        std::vector<float> end(out.begin() + 9600 + 33600, out.end());                  // 0.7 s after off
        CHECK(rms(mid) > 0.05, "note is still audible 100 ms into a 500 ms release");
        CHECK(peak(end) == 0.0, "silent after the release time has passed");
    }

    std::printf("Oscillator detune\n");
    {
        // Osc B on its own at +100 cents: one semitone up from A4, so 466.16 Hz.
        SynthCore* s = synth_create(kSR);
        synth_set_param(s, SYNTH_PARAM_OSC_A_LEVEL, 0.0f);
        synth_set_param(s, SYNTH_PARAM_OSC_B_LEVEL, 1.0f);
        synth_set_param(s, SYNTH_PARAM_OSC_B_DETUNE, 100.0f);
        synth_note_on(s, 69, 127);
        std::vector<float> out(kSecond);
        synth_render(s, out.data(), kSecond);
        synth_destroy(s);
        CHECK(std::fabs(estimate_hz(out, kSR) - 466.16) < 2.0, "+100 cents raises A4 to 466 Hz");
    }

    std::printf("Filter envelope opens then closes the filter\n");
    {
        // Low base cutoff (about 100 Hz) with the envelope fully on: the note starts
        // bright and falls back to dark after the decay.
        SynthCore* s = synth_create(kSR);
        synth_set_param(s, SYNTH_PARAM_OSC_A_WAVE, SYNTH_WAVE_SAW);
        synth_set_param(s, SYNTH_PARAM_FILTER_CUTOFF, cutoff_for_hz(100.0));
        synth_set_param(s, SYNTH_PARAM_FILTER_ENV_AMOUNT, 1.0f);
        synth_set_param(s, SYNTH_PARAM_FILTER_ENV_ATTACK, 0.0f);      // 1 ms
        synth_set_param(s, SYNTH_PARAM_FILTER_ENV_DECAY, 0.5f);       // about 100 ms
        synth_set_param(s, SYNTH_PARAM_FILTER_ENV_SUSTAIN, 0.0f);
        synth_note_on(s, 57, 127);  // A3, a saw with plenty of harmonics
        std::vector<float> out(kSecond);
        synth_render(s, out.data(), kSecond);
        synth_destroy(s);
        std::vector<float> early(out.begin() + 480, out.begin() + 1440);  // 10 - 30 ms
        std::vector<float> late(out.begin() + 24000, out.begin() + 28800);
        CHECK(rms(early) > rms(late) * 3.0, "filter sweep is brighter at the start of the note");
    }

    std::printf("LFO vibrato changes the pitch over time\n");
    {
        SynthCore* s = synth_create(kSR);
        synth_set_param(s, SYNTH_PARAM_LFO_RATE, 0.5f);       // 1 Hz
        synth_set_param(s, SYNTH_PARAM_LFO_PITCH_DEPTH, 100.0f);
        synth_note_on(s, 69, 127);
        std::vector<float> out(kSecond * 2);
        synth_render(s, out.data(), kSecond * 2);
        synth_destroy(s);
        CHECK(synth_lfo_hz(0.0f) > 0.049f && synth_lfo_hz(1.0f) > 19.9f, "LFO spans 0.05 to 20 Hz");
        CHECK(peak(out) < 1.0, "vibrato keeps output bounded");
        // Vibrato means the output is not the same as a plain 440 Hz sine.
        std::vector<float> plain = play_filtered(69, 1.0f, 0.0f);
        bool differs = false;
        for (size_t i = 0; i < kSecond; ++i) differs = differs || (out[i] != plain[i]);
        CHECK(differs, "vibrato changes the signal");
    }

    std::printf("Drive squares up the waveform\n");
    {
        double clean = plateau_fraction(play_a4(SYNTH_WAVE_SINE, 1.0f, SYNTH_WAVE_SINE, 0.0f), 0.45);
        SynthCore* s = synth_create(kSR);
        synth_set_param(s, SYNTH_PARAM_DRIVE, 1.0f);
        synth_note_on(s, 69, 127);
        std::vector<float> out(kSecond);
        synth_render(s, out.data(), kSecond);
        synth_destroy(s);
        CHECK(plateau_fraction(out, 0.45) > clean + 0.1, "full drive pushes the sine toward a square");
    }

    std::printf("Reverb adds a tail after the note\n");
    {
        // Hold 0.2 s, release, then measure 0.5 s later.
        auto run = [](float mix) {
            SynthCore* s = synth_create(kSR);
            synth_set_param(s, SYNTH_PARAM_REVERB_MIX, mix);
            synth_set_param(s, SYNTH_PARAM_REVERB_SIZE, 0.8f);
            synth_note_on(s, 69, 127);
            std::vector<float> out(9600 + 24000);
            synth_render(s, out.data(), 9600);
            synth_note_off(s, 69);
            synth_render(s, out.data() + 9600, 24000);
            synth_destroy(s);
            return out;
        };
        std::vector<float> dry = run(0.0f);
        std::vector<float> wet = run(0.5f);
        std::vector<float> dryTail(dry.end() - 4800, dry.end());
        std::vector<float> wetTail(wet.end() - 4800, wet.end());
        CHECK(peak(dryTail) == 0.0, "no reverb: silent after release");
        CHECK(rms(wetTail) > 0.001, "reverb: the tail keeps ringing after release");
        bool finite = true;
        for (float s : wet) finite = finite && std::isfinite(s);
        CHECK(finite, "reverb output is finite");
    }

    std::printf("Parameter schema\n");
    {
        CHECK(synth_param_count() == SYNTH_PARAM_COUNT, "param count matches enum");
        SynthParamInfo a = synth_param_info(SYNTH_PARAM_OSC_A_WAVE);
        CHECK(std::strcmp(a.id, "oscA_wave") == 0, "osc A wave id");
        CHECK(a.kind == SYNTH_PARAM_WAVEFORM, "osc A wave is a waveform choice");
        CHECK(a.min == 0 && a.max == SYNTH_WAVE_COUNT - 1, "waveform range covers all choices");
        CHECK(a.defaultValue == SYNTH_WAVE_SINE, "osc A defaults to sine");
        SynthParamInfo b = synth_param_info(SYNTH_PARAM_OSC_B_LEVEL);
        CHECK(b.defaultValue == 0.0f, "osc B starts silent (level 0)");
        CHECK(synth_param_info(999).id[0] == '\0', "out-of-range index gives empty info");
        CHECK(synth_waveform_count() == 4, "four waveform choices");
        CHECK(std::strcmp(synth_waveform_name(SYNTH_WAVE_SQUARE), "Square") == 0,
              "waveform name lookup");
    }

    std::printf("Set/get clamps to range\n");
    {
        SynthCore* s = synth_create(kSR);
        synth_set_param(s, SYNTH_PARAM_OSC_A_LEVEL, 5.0f);
        CHECK(synth_get_param(s, SYNTH_PARAM_OSC_A_LEVEL) == 1.0f, "level clamps to 1");
        synth_set_param(s, SYNTH_PARAM_OSC_A_WAVE, -3.0f);
        CHECK(synth_get_param(s, SYNTH_PARAM_OSC_A_WAVE) == 0.0f, "waveform clamps to 0");
        synth_destroy(s);
    }

    std::printf("Silence before any note\n");
    {
        SynthCore* s = synth_create(kSR);
        std::vector<float> out(kSecond);
        synth_render(s, out.data(), kSecond);
        CHECK(peak(out) == 0.0, "silent with no input");
        synth_destroy(s);
    }

    std::printf("Default sine: A4 gives a 440 Hz tone\n");
    {
        SynthCore* s = synth_create(kSR);
        synth_note_on(s, 69, 100);
        std::vector<float> out(kSecond);
        synth_render(s, out.data(), kSecond);
        CHECK(peak(out) > 0.1, "output is audible");
        CHECK(std::fabs(estimate_hz(out, kSR) - 440.0) < 2.0, "frequency is 440 Hz");
        synth_destroy(s);
    }

    std::printf("Every waveform is audible at 440 Hz\n");
    for (int w = 0; w < SYNTH_WAVE_COUNT; ++w) {
        std::vector<float> out = play_a4(w, 1.0f, SYNTH_WAVE_SINE, 0.0f);
        char msg[96];
        std::snprintf(msg, sizeof msg, "%s: audible and 440 Hz",
                      synth_waveform_name(w));
        CHECK(peak(out) > 0.1 && std::fabs(estimate_hz(out, kSR) - 440.0) < 2.0, msg);
    }

    std::printf("Waveform shapes\n");
    {
        // Amplitude is 0.5. Uniform-distributed waves (saw, triangle) have
        // RMS 0.5/sqrt(3); sine has 0.5/sqrt(2); square has 0.5.
        double sineRms = rms(play_a4(SYNTH_WAVE_SINE, 1.0f, SYNTH_WAVE_SINE, 0.0f));
        double sawRms = rms(play_a4(SYNTH_WAVE_SAW, 1.0f, SYNTH_WAVE_SINE, 0.0f));
        double triRms = rms(play_a4(SYNTH_WAVE_TRIANGLE, 1.0f, SYNTH_WAVE_SINE, 0.0f));
        double sqRms = rms(play_a4(SYNTH_WAVE_SQUARE, 1.0f, SYNTH_WAVE_SINE, 0.0f));
        CHECK(std::fabs(sineRms - 0.5 / std::sqrt(2.0)) < 0.01, "sine RMS is 0.354");
        CHECK(std::fabs(sawRms - 0.5 / std::sqrt(3.0)) < 0.01, "saw RMS is 0.289");
        CHECK(std::fabs(triRms - 0.5 / std::sqrt(3.0)) < 0.01, "triangle RMS is 0.289");
        CHECK(std::fabs(sqRms - 0.5) < 0.01, "square RMS is 0.5");

        double sqPlateau = plateau_fraction(play_a4(SYNTH_WAVE_SQUARE, 1.0f, SYNTH_WAVE_SINE, 0.0f), 0.45);
        double sinPlateau = plateau_fraction(play_a4(SYNTH_WAVE_SINE, 1.0f, SYNTH_WAVE_SINE, 0.0f), 0.45);
        CHECK(sqPlateau > 0.95, "square spends almost all time at its two levels");
        CHECK(sinPlateau < 0.5, "sine does not sit at its peaks");
    }

    std::printf("Mixer levels\n");
    {
        double full = peak(play_a4(SYNTH_WAVE_SINE, 1.0f, SYNTH_WAVE_SINE, 0.0f));
        double half = peak(play_a4(SYNTH_WAVE_SINE, 0.5f, SYNTH_WAVE_SINE, 0.0f));
        CHECK(std::fabs(half - full * 0.5) < 0.005, "level 0.5 halves the amplitude");

        std::vector<float> silent = play_a4(SYNTH_WAVE_SINE, 0.0f, SYNTH_WAVE_SINE, 0.0f);
        CHECK(peak(silent) == 0.0, "osc A level 0 with osc B at 0 is silent");

        std::vector<float> bOnly = play_a4(SYNTH_WAVE_SINE, 0.0f, SYNTH_WAVE_SQUARE, 1.0f);
        CHECK(std::fabs(rms(bOnly) - 0.5) < 0.01, "osc B alone plays its own waveform");
    }

    std::printf("Mixer is linear: mix(A,B) == A + B\n");
    {
        std::vector<float> a = play_a4(SYNTH_WAVE_SAW, 0.6f, SYNTH_WAVE_SINE, 0.0f);
        std::vector<float> b = play_a4(SYNTH_WAVE_SQUARE, 0.0f, SYNTH_WAVE_SQUARE, 0.4f);
        std::vector<float> mix = play_a4(SYNTH_WAVE_SAW, 0.6f, SYNTH_WAVE_SQUARE, 0.4f);
        double worst = 0.0;
        for (size_t i = 0; i < mix.size(); ++i) {
            worst = std::fmax(worst, std::fabs(mix[i] - (a[i] + b[i])));
        }
        CHECK(worst < 1e-5, "mixed output equals the sum of the parts");
    }

    std::printf("Note off returns to silence\n");
    {
        SynthCore* s = synth_create(kSR);
        synth_note_on(s, 60, 127);
        std::vector<float> warm(4800);
        synth_render(s, warm.data(), 4800);
        synth_note_off(s, 60);
        std::vector<float> tail(4800);  // 100 ms, longer than the fade
        synth_render(s, tail.data(), 4800);
        std::vector<float> after(4800);
        synth_render(s, after.data(), 4800);
        CHECK(peak(after) == 0.0, "silent after release");
        synth_destroy(s);
    }

    std::printf("Chords: several notes sound at once\n");
    {
        SynthCore* one = synth_create(kSR);
        synth_note_on(one, 60, 100);
        std::vector<float> single(kSecond / 2);
        synth_render(one, single.data(), kSecond / 2);
        synth_destroy(one);

        SynthCore* chord = synth_create(kSR);
        synth_note_on(chord, 60, 100);
        synth_note_on(chord, 64, 100);
        synth_note_on(chord, 67, 100);
        std::vector<float> triad(kSecond / 2);
        synth_render(chord, triad.data(), kSecond / 2);
        CHECK(rms(triad) > rms(single) * 1.2, "a triad is louder than one note");

        // Release the root: the other two keep sounding.
        synth_note_off(chord, 60);
        std::vector<float> rest(kSecond / 2);
        synth_render(chord, rest.data(), kSecond / 2);
        CHECK(peak(rest) > 0.05, "releasing one chord note keeps the others sounding");

        synth_note_off(chord, 64);
        synth_note_off(chord, 67);
        synth_render(chord, rest.data(), kSecond / 2);  // let the release finish
        std::vector<float> after(4800);
        synth_render(chord, after.data(), 4800);
        CHECK(peak(after) == 0.0, "silent once every chord note is released");
        synth_destroy(chord);
    }

    std::printf("Chords: a repeated note releases one voice at a time\n");
    {
        SynthCore* s = synth_create(kSR);
        synth_note_on(s, 69, 100);
        synth_note_on(s, 69, 100);
        synth_note_off(s, 69);
        std::vector<float> out(kSecond / 2);
        synth_render(s, out.data(), kSecond / 2);
        CHECK(peak(out) > 0.05, "one of two held A4s still sounds");
        synth_destroy(s);
    }

    std::printf("Chords: more notes than voices steal without failing\n");
    {
        SynthCore* s = synth_create(kSR);
        for (int n = 40; n < 52; ++n) synth_note_on(s, n, 100);  // 12 notes, 8 voices
        std::vector<float> out(kSecond / 4);
        synth_render(s, out.data(), kSecond / 4);
        bool finite = true;
        for (float x : out) if (!std::isfinite(x)) finite = false;
        CHECK(finite && peak(out) > 0.05, "12 held notes render finite, audible output");
        synth_destroy(s);
    }

    std::printf("Deterministic: same input gives identical output\n");
    {
        std::vector<float> a(kSecond), b(kSecond);
        for (auto* buf : {&a, &b}) {
            SynthCore* s = synth_create(kSR);
            synth_set_param(s, SYNTH_PARAM_OSC_A_WAVE, SYNTH_WAVE_SAW);
            synth_set_param(s, SYNTH_PARAM_OSC_B_LEVEL, 0.5f);
            synth_note_on(s, 72, 90);
            synth_render(s, buf->data(), kSecond);
            synth_destroy(s);
        }
        CHECK(a == b, "two renders are bit-identical");
    }

    std::printf("Neural source: timbre sets the harmonics\n");
    {
        CHECK(std::strcmp(synth_param_info(SYNTH_PARAM_SOURCE_MIX).id, "source_mix") == 0,
              "source mix id");
        CHECK(synth_param_info(SYNTH_PARAM_SOURCE_MIX).defaultValue == 0.0f,
              "source mix defaults to oscillators only");

        // A4 with mix 1 and only a sine fundamental: pitch is 440 Hz.
        const float sine[1] = {1.0f};
        SynthCore* s = synth_create(kSR);
        synth_set_timbre(s, sine, 1);
        synth_set_param(s, SYNTH_PARAM_SOURCE_MIX, 1.0f);
        synth_note_on(s, 69, 127);
        std::vector<float> out(kSecond);
        synth_render(s, out.data(), kSecond);
        synth_destroy(s);
        const double hz = estimate_hz(out, kSR);
        CHECK(std::fabs(hz - 440.0) < 4.4, "sine timbre plays A4 at 440 Hz (within 1%)");

        // Only harmonic 2 present: the waveform crosses zero twice per cycle of
        // the fundamental, so A4 should read as 880 Hz.
        const float second[2] = {0.0f, 1.0f};
        s = synth_create(kSR);
        synth_set_timbre(s, second, 2);
        synth_set_param(s, SYNTH_PARAM_SOURCE_MIX, 1.0f);
        synth_note_on(s, 69, 127);
        synth_render(s, out.data(), kSecond);
        synth_destroy(s);
        CHECK(std::fabs(estimate_hz(out, kSR) - 880.0) < 8.8,
              "harmonic 2 alone plays one octave up (880 Hz)");
    }

    std::printf("Neural source: default mix leaves the sound unchanged\n");
    {
        const float sine[1] = {1.0f};
        std::vector<float> plain(kSecond / 2), with(kSecond / 2);
        for (auto* buf : {&plain, &with}) {
            SynthCore* s = synth_create(kSR);
            if (buf == &with) synth_set_timbre(s, sine, 1);  // loaded, but mix stays 0
            synth_note_on(s, 69, 100);
            synth_render(s, buf->data(), kSecond / 2);
            synth_destroy(s);
        }
        CHECK(plain == with, "timbre loaded at mix 0 changes nothing");
    }

    std::printf("Neural source: stays on pitch across the keyboard\n");
    {
        const float sine[1] = {1.0f};
        for (int note : {21, 108}) {  // lowest and highest piano keys
            SynthCore* s = synth_create(kSR);
            synth_set_timbre(s, sine, 1);
            synth_set_param(s, SYNTH_PARAM_SOURCE_MIX, 1.0f);
            synth_note_on(s, note, 127);
            // Eight seconds: a one-second window cannot resolve 27.5 Hz to within 1%.
            std::vector<float> out(8 * kSecond);
            synth_render(s, out.data(), 8 * kSecond);
            synth_destroy(s);
            const double want = 440.0 * std::pow(2.0, (note - 69) / 12.0);
            const double got = estimate_hz(out, kSR);
            const bool ok = std::fabs(got - want) < want * 0.01;
            std::printf("  %s MIDI %d: want %.1f Hz, got %.1f Hz\n", ok ? "ok  " : "FAIL",
                        note, want, got);
            if (!ok) ++failures;
        }
    }

    std::printf("Neural source: loudness envelope shapes the note\n");
    {
        // A sine rising from silence to full level over the 4 s note.
        const float sine[1] = {1.0f};
        const float rise[2] = {0.0f, 1.0f};
        SynthCore* s = synth_create(kSR);
        synth_set_timbre_shaped(s, sine, 1, rise, 2);
        synth_set_param(s, SYNTH_PARAM_SOURCE_MIX, 1.0f);
        synth_note_on(s, 69, 127);
        std::vector<float> out(4 * kSecond);
        synth_render(s, out.data(), 4 * kSecond);
        synth_destroy(s);
        auto peakOf = [&](int from, int to) {
            float p = 0.0f;
            for (int i = from; i < to; ++i) p = std::fmax(p, std::fabs(out[i]));
            return p;
        };
        const float early = peakOf(0, kSecond / 2);
        const float late = peakOf(7 * kSecond / 2, 4 * kSecond);
        CHECK(late > 4.0f * early, "envelope rising 0 to 1 makes the late note much louder");

        // The same timbre without an envelope stays at full level from the start.
        s = synth_create(kSR);
        synth_set_timbre(s, sine, 1);
        synth_set_param(s, SYNTH_PARAM_SOURCE_MIX, 1.0f);
        synth_note_on(s, 69, 127);
        synth_render(s, out.data(), 4 * kSecond);
        synth_destroy(s);
        CHECK(peakOf(0, kSecond / 2) > 0.5f * peakOf(7 * kSecond / 2, 4 * kSecond),
              "a timbre loaded without an envelope has no level change");
    }

    std::printf("Neural source: clearing and sample-rate change\n");
    {
        const float sine[1] = {1.0f};
        SynthCore* s = synth_create(kSR);
        synth_set_timbre(s, sine, 1);
        synth_set_param(s, SYNTH_PARAM_SOURCE_MIX, 1.0f);
        synth_set_timbre(s, nullptr, 0);  // clear
        synth_note_on(s, 69, 127);
        std::vector<float> cleared(kSecond / 4);
        synth_render(s, cleared.data(), kSecond / 4);
        CHECK(peak(cleared) < 1e-6, "cleared timbre with mix 1 is silent");

        synth_set_timbre(s, sine, 1);
        synth_set_sample_rate(s, 44100.0);  // rebuilds the loaded timbre
        std::vector<float> after(kSecond / 4);
        synth_render(s, after.data(), kSecond / 4);
        CHECK(peak(after) > 0.05, "timbre still sounds after a sample-rate change");
        synth_destroy(s);
    }

    std::printf("Neural source: movement (step 2c)\n");
    {
        const int kFrameSamples = static_cast<int>(kSR / SYNTH_MOVEMENT_RATE_HZ);  // 960 at 48 kHz
        const int kFrames = 50;
        const float sine[1] = {1.0f};
        float saw[32];
        for (int k = 0; k < 32; ++k) saw[k] = 1.0f / static_cast<float>(k + 1);

        // Level of the sound in the middle half of one 20 ms frame.
        auto frame_rms = [kFrameSamples](const std::vector<float>& x, int frame) {
            const int a = frame * kFrameSamples + kFrameSamples / 4;
            const int b = frame * kFrameSamples + 3 * kFrameSamples / 4;
            double acc = 0.0;
            for (int i = a; i < b; ++i) acc += static_cast<double>(x[i]) * x[i];
            return std::sqrt(acc / (b - a));
        };
        auto render = [](SynthCore* s, int frames) {
            std::vector<float> out(frames);
            synth_render(s, out.data(), frames);
            return out;
        };

        // Loop and wrap: frames 0-9 play once, frames 10-29 loop, frames 30-49 are never played.
        std::vector<float> amp(kFrames), zeros(kFrames, 0.0f);
        for (int k = 0; k < kFrames; ++k) amp[k] = k < 10 ? 1.0f : (k < 30 ? 0.25f : 0.75f);
        SynthMovement loop = {kFrames, 10, 30, amp.data(), zeros.data(), zeros.data()};
        SynthCore* s = synth_create(kSR);
        synth_set_timbre_moving(s, sine, 1, &loop);
        synth_set_param(s, SYNTH_PARAM_SOURCE_MIX, 1.0f);
        synth_note_on(s, 69, 127);
        const std::vector<float> looped = render(s, 2 * kSecond);
        synth_destroy(s);
        const double r5 = frame_rms(looped, 5), r25 = frame_rms(looped, 25);
        const double r35 = frame_rms(looped, 35), r45 = frame_rms(looped, 45);
        CHECK(std::fabs(r25 / r5 - 0.25) < 0.03, "amp track sets the level: frame 25 is a quarter of frame 5");
        CHECK(std::fabs(r35 / r25 - 1.0) < 0.03, "playback wraps to loopStart after loopEnd (frame 35 repeats frame 25)");
        CHECK(std::fabs(r45 / r25 - 1.0) < 0.03, "the loop keeps repeating (frame 45 repeats frame 25)");
        CHECK(r35 / r5 < 0.35, "frames after loopEnd (level 0.75) are never played");

        // Pitch track: +100 cents raises A4 to 466 Hz. Movement 0 leaves it at 440 Hz.
        std::vector<float> ones(kFrames, 1.0f), cents(kFrames, 100.0f);
        SynthMovement pitched = {kFrames, 0, kFrames, ones.data(), cents.data(), zeros.data()};
        s = synth_create(kSR);
        synth_set_timbre_moving(s, sine, 1, &pitched);
        synth_set_param(s, SYNTH_PARAM_SOURCE_MIX, 1.0f);
        synth_note_on(s, 69, 127);
        const std::vector<float> up = render(s, kSecond);
        CHECK(std::fabs(estimate_hz(up, kSR) - 466.16) < 2.0, "pitch track +100 cents raises A4 to 466 Hz");
        synth_destroy(s);
        s = synth_create(kSR);
        synth_set_timbre_moving(s, sine, 1, &pitched);
        synth_set_param(s, SYNTH_PARAM_SOURCE_MIX, 1.0f);
        synth_set_param(s, SYNTH_PARAM_MOVEMENT, 0.0f);
        synth_note_on(s, 69, 127);
        const std::vector<float> still = render(s, kSecond);
        CHECK(std::fabs(estimate_hz(still, kSR) - 440.0) < 1.0, "movement 0 ignores the pitch track (440 Hz)");
        synth_destroy(s);

        // Brightness track: -4 octaves closes the filter, so a saw loses its upper harmonics.
        std::vector<float> dark(kFrames, -4.0f);
        SynthMovement brightTrack = {kFrames, 0, kFrames, ones.data(), zeros.data(), dark.data()};
        s = synth_create(kSR);
        synth_set_timbre_moving(s, saw, 32, &brightTrack);
        synth_set_param(s, SYNTH_PARAM_SOURCE_MIX, 1.0f);
        synth_note_on(s, 69, 127);
        const std::vector<float> closed = render(s, kSecond);
        synth_destroy(s);
        s = synth_create(kSR);
        synth_set_timbre(s, saw, 32);
        synth_set_param(s, SYNTH_PARAM_SOURCE_MIX, 1.0f);
        synth_note_on(s, 69, 127);
        const std::vector<float> open = render(s, kSecond);
        synth_destroy(s);
        CHECK(rms(closed) < rms(open) * 0.95, "brightness track -4 octaves lowers the filtered level");

        // Movement 0 is the same as no movement: any tracks, bit for bit.
        std::vector<float> quiet(kFrames, 0.3f), sharp(kFrames, 200.0f), dim(kFrames, -2.0f);
        SynthMovement busy = {kFrames, 5, 40, quiet.data(), sharp.data(), dim.data()};
        SynthCore* withTracks = synth_create(kSR);
        synth_set_timbre_moving(withTracks, saw, 32, &busy);
        synth_set_param(withTracks, SYNTH_PARAM_SOURCE_MIX, 1.0f);
        synth_set_param(withTracks, SYNTH_PARAM_MOVEMENT, 0.0f);
        SynthCore* plain = synth_create(kSR);
        synth_set_timbre(plain, saw, 32);
        synth_set_param(plain, SYNTH_PARAM_SOURCE_MIX, 1.0f);
        synth_note_on(withTracks, 69, 127);
        synth_note_on(plain, 69, 127);
        const std::vector<float> a = render(withTracks, kSecond / 2);
        const std::vector<float> b = render(plain, kSecond / 2);
        CHECK(a == b, "movement 0 is bit-identical to a timbre without movement");

        // A timbre without movement ignores the Movement parameter.
        synth_destroy(withTracks);
        synth_destroy(plain);
        SynthCore* noMove = synth_create(kSR);
        synth_set_timbre(noMove, saw, 32);
        synth_set_param(noMove, SYNTH_PARAM_SOURCE_MIX, 1.0f);
        synth_set_param(noMove, SYNTH_PARAM_MOVEMENT, 0.0f);
        synth_note_on(noMove, 69, 127);
        const std::vector<float> c = render(noMove, kSecond / 2);
        synth_destroy(noMove);
        SynthCore* noMove1 = synth_create(kSR);
        synth_set_timbre(noMove1, saw, 32);
        synth_set_param(noMove1, SYNTH_PARAM_SOURCE_MIX, 1.0f);
        synth_note_on(noMove1, 69, 127);
        const std::vector<float> d = render(noMove1, kSecond / 2);
        synth_destroy(noMove1);
        CHECK(c == d, "a timbre without movement sounds the same at any Movement setting");

        // Movement survives a sample-rate change...
        SynthCore* moved = synth_create(kSR);
        synth_set_timbre_moving(moved, saw, 32, &busy);
        synth_set_param(moved, SYNTH_PARAM_SOURCE_MIX, 1.0f);
        synth_set_sample_rate(moved, kSR);
        synth_note_on(moved, 69, 127);
        const std::vector<float> kept = render(moved, kSecond / 2);
        CHECK(kept != b, "movement survives a sample-rate change (differs from no movement)");
        synth_destroy(moved);

        // ...and a plain timbre load clears it.
        SynthCore* cleared = synth_create(kSR);
        synth_set_timbre_moving(cleared, saw, 32, &busy);
        synth_set_timbre(cleared, saw, 32);
        synth_set_param(cleared, SYNTH_PARAM_SOURCE_MIX, 1.0f);
        synth_note_on(cleared, 69, 127);
        const std::vector<float> wiped = render(cleared, kSecond / 2);
        synth_destroy(cleared);
        CHECK(wiped == b, "loading a timbre without movement clears the movement");
    }

    std::printf("Reset: a reset core renders like a fresh one\n");
    {
        // Fresh core: one note with vibrato and reverb.
        SynthCore* fresh = synth_create(kSR);
        synth_set_param(fresh, SYNTH_PARAM_LFO_PITCH_DEPTH, 50.0f);
        synth_set_param(fresh, SYNTH_PARAM_REVERB_MIX, 0.5f);
        synth_set_param(fresh, SYNTH_PARAM_REVERB_SIZE, 1.0f);
        synth_note_on(fresh, 69, 100);
        std::vector<float> a(kSecond / 2);
        synth_render(fresh, a.data(), kSecond / 2);
        synth_destroy(fresh);

        // Dirty core: loud chord, reverb tail, LFO advanced; then reset and the same note.
        SynthCore* dirty = synth_create(kSR);
        synth_set_param(dirty, SYNTH_PARAM_LFO_PITCH_DEPTH, 50.0f);
        synth_set_param(dirty, SYNTH_PARAM_REVERB_MIX, 0.5f);
        synth_set_param(dirty, SYNTH_PARAM_REVERB_SIZE, 1.0f);
        for (int n : {60, 64, 67, 72}) synth_note_on(dirty, n, 127);
        std::vector<float> junk(kSecond);
        synth_render(dirty, junk.data(), kSecond);
        synth_note_off(dirty, 60);
        synth_note_off(dirty, 64);
        synth_note_off(dirty, 67);
        synth_note_off(dirty, 72);
        synth_render(dirty, junk.data(), kSecond / 2);  // let the reverb ring
        synth_reset(dirty);
        synth_note_on(dirty, 69, 100);
        std::vector<float> b(kSecond / 2);
        synth_render(dirty, b.data(), kSecond / 2);
        synth_destroy(dirty);
        CHECK(a == b, "after reset, the same note is bit-identical to a fresh core");
        CHECK(peak(a) > 0.01, "the reset comparison note is audible");

        // Reset keeps parameters and the loaded timbre: a sine timbre still plays at 440 Hz.
        const float sine[1] = {1.0f};
        SynthCore* kept = synth_create(kSR);
        synth_set_timbre(kept, sine, 1);
        synth_set_param(kept, SYNTH_PARAM_SOURCE_MIX, 1.0f);
        synth_reset(kept);
        synth_note_on(kept, 69, 127);
        std::vector<float> tone(kSecond);
        synth_render(kept, tone.data(), kSecond);
        synth_destroy(kept);
        CHECK(std::fabs(estimate_hz(tone, kSR) - 440.0) < 4.4, "reset keeps the loaded timbre");
    }

    std::printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "PASSED",
                failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
