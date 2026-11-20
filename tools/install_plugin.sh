#!/bin/sh
# Install the Tracewright plugin into KiCad 10 (per-user) on macOS or Linux.
#   tools/install_plugin.sh [path/to/tracewright] [kicad-version]
# Copies the plugin executable, manifest and icons into KiCad's user plugins folder:
#   macOS: ~/Documents/KiCad/<version>/plugins/com.tracewright.router/
#   Linux: ${XDG_DATA_HOME:-~/.local/share}/kicad/<version>/plugins/com.tracewright.router/
# ($KICAD_DOCUMENTS_HOME replaces ~/Documents or the XDG data folder, as it does for KiCad.)
# Restart KiCad's PCB editor afterwards; the "Tracewright: Autoroute" button appears in its
# toolbar (Preferences > Plugins must have "Enable KiCad API" turned on).
set -eu
root=$(cd "$(dirname "$0")/.." && pwd)
exe=${1:-"$root/target/release/tracewright"}
version=${2:-10.0}
[ -f "$exe" ] || { echo "plugin executable not found: $exe (build with: cargo build --release)" >&2; exit 1; }
case $(uname -s) in
    Darwin) docs=${KICAD_DOCUMENTS_HOME:-"$HOME/Documents"}/KiCad ;;
    *) docs=${KICAD_DOCUMENTS_HOME:-${XDG_DATA_HOME:-"$HOME/.local/share"}}/kicad ;;
esac
dest="$docs/$version/plugins/com.tracewright.router"
mkdir -p "$dest/icons"
cp "$exe" "$dest/tracewright"
chmod 755 "$dest/tracewright"
cp "$root"/plugin/icons/*.png "$dest/icons/"
# KiCad skips actions whose entrypoint is an absolute path, so keep it relative to the manifest.
sed 's/"entrypoint": *"tracewright\.exe"/"entrypoint": "tracewright"/' "$root/plugin/plugin.json" > "$dest/plugin.json"
grep -q '"entrypoint": "tracewright"' "$dest/plugin.json" || { echo "could not set the entrypoint in $dest/plugin.json" >&2; exit 1; }
if [ "$(uname -s)" = Darwin ]; then
    # A downloaded (not locally built) binary carries Gatekeeper's quarantine flag, which would
    # block KiCad from launching it.
    xattr -d com.apple.quarantine "$dest/tracewright" 2>/dev/null || true
fi
echo "Installed to $dest"
