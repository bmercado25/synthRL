#!/bin/bash
# Installs the AUv3 build into ~/Applications so hosts such as GarageBand can
# find it, then registers the extension with the system.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/build/SynthHost.app"
DEST="$HOME/Applications/SynthRL.app"

if [ ! -d "$SRC" ]; then
    echo "Missing $SRC. Run scripts/build.sh first." >&2
    exit 1
fi

mkdir -p "$HOME/Applications"
rm -rf "$DEST"
cp -R "$SRC" "$DEST"
codesign --force --sign - --timestamp=none --entitlements "$ROOT/plugin/Extension.entitlements" "$DEST/Contents/PlugIns/SynthExtension.appex"
codesign --force --sign - --timestamp=none "$DEST"

# Re-registering a replaced bundle can drop the entry, so register, enable, then
# restart the component registry so hosts see the new copy.
pluginkit -a "$DEST/Contents/PlugIns/SynthExtension.appex"
pluginkit -e use -i com.synthrl.host.synth || true
killall -9 AudioComponentRegistrar 2>/dev/null || true
sleep 3
echo "==> Installed $DEST"
pluginkit -m -v -i com.synthrl.host.synth || true
