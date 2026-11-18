// Prints the parameter schema from core/synth_core.h as JSON.
// The Python side reads schema/synth_params.json, which is generated from this
// tool. scripts/test.sh regenerates it and fails if the committed copy is stale.
#include "synth_core.h"

#include <cstdio>

int main() {
    const int count = synth_param_count();
    std::printf("{\n  \"schema_version\": 1,\n  \"parameters\": [\n");
    for (int i = 0; i < count; ++i) {
        const SynthParamInfo p = synth_param_info(i);
        std::printf("    {\"index\": %d, \"id\": \"%s\", \"name\": \"%s\", "
                    "\"type\": \"%s\", \"min\": %.6g, \"max\": %.6g, \"default\": %.6g}%s\n",
                    i, p.id, p.name,
                    p.kind == SYNTH_PARAM_WAVEFORM ? "waveform" : "continuous",
                    p.min, p.max, p.defaultValue,
                    i + 1 < count ? "," : "");
    }
    std::printf("  ]\n}\n");
    return 0;
}
