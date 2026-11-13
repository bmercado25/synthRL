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
        CHECK(synth_param_count() == 10, "ten parameters: osc, filter and envelope");
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

    std::printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "PASSED",
                failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
