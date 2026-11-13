// Core-level tests: MIDI-style calls in, samples out. No Apple APIs.
#include "../core/synth_core.h"

#include <cmath>
#include <cstdio>
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

int main() {
    const double sr = 48000.0;
    const int frames = 48000;  // one second

    std::printf("Silence before any note\n");
    {
        SynthCore* s = synth_create(sr);
        std::vector<float> out(frames);
        synth_render(s, out.data(), frames);
        CHECK(peak(out) == 0.0, "silent with no input");
        synth_destroy(s);
    }

    std::printf("Note on A4 (69) produces a 440 Hz tone\n");
    {
        SynthCore* s = synth_create(sr);
        synth_note_on(s, 69, 100);
        std::vector<float> out(frames);
        synth_render(s, out.data(), frames);
        double hz = estimate_hz(out, sr);
        CHECK(peak(out) > 0.1, "output is audible");
        CHECK(std::fabs(hz - 440.0) < 2.0, "frequency is 440 Hz");
        synth_destroy(s);
    }

    std::printf("Note off returns to silence\n");
    {
        SynthCore* s = synth_create(sr);
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
        std::vector<float> a(frames), b(frames);
        for (auto* buf : {&a, &b}) {
            SynthCore* s = synth_create(sr);
            synth_note_on(s, 72, 90);
            synth_render(s, buf->data(), frames);
            synth_destroy(s);
        }
        CHECK(a == b, "two renders are bit-identical");
    }

    std::printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "PASSED",
                failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
