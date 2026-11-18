#!/bin/bash
# Full test pipeline:
#   1. core unit tests (C++, no Apple APIs)
#   2. build the AUv3 extension and container app
#   3. AUv3 integration test: instantiate the unit, send MIDI, check audio out
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "### 1/3 Core tests"
clang++ -std=c++17 -O2 -Wall -Wextra core/synth_core.cpp tests/core_test.cpp -o build/core_test
./build/core_test

# The parameter schema is the single source of truth. The committed JSON that
# the Python side reads must match what the core currently declares.
clang++ -std=c++17 -O2 -Wall -Wextra -Icore core/synth_core.cpp tools/export_schema.cpp -o build/export_schema
./build/export_schema > build/synth_params.json
diff -u schema/synth_params.json build/synth_params.json || {
    echo "schema/synth_params.json is stale. Regenerate with:"
    echo "  ./build/export_schema > schema/synth_params.json"
    exit 1
}
echo "parameter schema matches schema/synth_params.json"

# Python side of the neural source (analysis, timbre files). Needs the venv
# from CLAUDE.md ("Python environment"); skipped loudly if it is missing.
echo
echo "### Python source pipeline tests"
if [ -x .venv/bin/python ]; then
    .venv/bin/python -m pytest neural/tests -q
else
    echo "SKIPPED: no .venv. Create it as described in CLAUDE.md"
fi

echo
echo "### 2/3 Build AUv3"
./scripts/build.sh > build/build.log 2>&1 || { cat build/build.log; exit 1; }
echo "built (log: build/build.log)"

echo
echo "### 3/3 AUv3 integration test"
swiftc -O -module-name AUTest \
    -import-objc-header plugin/Bridging-Header.h \
    plugin/SynthAudioUnit.swift tests/au/main.swift build/synth_core.o \
    -lc++ -framework AudioToolbox -framework AVFAudio -framework Foundation \
    -o build/au_test
# The installed copy in ~/Applications is registered with the system, which makes
# the in-process test try to launch it out of process (4097). Ignore it for the
# duration of the test, then restore it.
if pluginkit -m -i com.synthrl.host.synth >/dev/null 2>&1; then
    pluginkit -e ignore -i com.synthrl.host.synth || true
    trap 'pluginkit -e use -i com.synthrl.host.synth || true' EXIT
fi
./build/au_test build/au_output.wav
