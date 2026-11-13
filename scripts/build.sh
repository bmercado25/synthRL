#!/bin/bash
# Builds the portable core, the AUv3 extension (standalone, not embedded in the
# host) and the SwiftUI host app that runs the synth in-process.
# Uses only Command Line Tools (no full Xcode required).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/build"
APP="$BUILD/SynthHost.app"
APPEX="$BUILD/SynthExtension.appex"

rm -rf "$APP" "$APPEX" "$BUILD/synth_core.o"
mkdir -p "$BUILD"

echo "==> Core (C++)"
clang++ -std=c++17 -O2 -c "$ROOT/core/synth_core.cpp" -o "$BUILD/synth_core.o"

echo "==> AUv3 extension (Swift)"
mkdir -p "$APPEX/Contents/MacOS"
cp "$ROOT/plugin/Info.plist" "$APPEX/Contents/Info.plist"
swiftc -O -parse-as-library \
    -module-name SynthExtension \
    -import-objc-header "$ROOT/plugin/Bridging-Header.h" \
    -Xlinker -application_extension \
    -Xlinker -e -Xlinker _NSExtensionMain \
    -framework AudioToolbox -framework AVFAudio -framework Foundation \
    -framework SwiftUI -framework AppKit -framework Combine -framework CoreAudioKit \
    -lc++ \
    "$ROOT/plugin/SynthAudioUnit.swift" "$ROOT/plugin/SynthViewController.swift" \
    "$ROOT/ui/SynthUI.swift" "$BUILD/synth_core.o" \
    -o "$APPEX/Contents/MacOS/SynthExtension"

echo "==> Host app (SwiftUI window, loads the AU in-process)"
mkdir -p "$APP/Contents/MacOS"
cp "$ROOT/host/Info.plist" "$APP/Contents/Info.plist"
swiftc -O -parse-as-library \
    -module-name SynthHost \
    -import-objc-header "$ROOT/plugin/Bridging-Header.h" \
    "$ROOT/host/SynthHostApp.swift" "$ROOT/ui/SynthUI.swift" "$ROOT/plugin/SynthAudioUnit.swift" "$BUILD/synth_core.o" \
    -lc++ \
    -framework SwiftUI -framework AppKit -framework AudioToolbox \
    -framework AVFAudio -framework Foundation -framework Combine \
    -o "$APP/Contents/MacOS/SynthHost"

echo "==> Embedding the AU extension in the containing app (AUv3 layout)"
mkdir -p "$APP/Contents/PlugIns"
cp -R "$APPEX" "$APP/Contents/PlugIns/"

echo "==> Ad-hoc code signing (extension first, then the containing app)"
codesign --force --sign - --timestamp=none \
    --entitlements "$ROOT/plugin/Extension.entitlements" "$APP/Contents/PlugIns/SynthExtension.appex"
# Host is unsandboxed: a sandboxed host cannot load the AU in-process (-3000).
codesign --force --sign - --timestamp=none "$APP"
codesign --verify --deep --strict "$APP"

echo "==> Built $APP and $APPEX"
