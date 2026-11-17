#!/usr/bin/env bash
# Adds ∞ TAPEDECK to the app menu (and, with --autostart, to login).
# The tray needs a Python with PySide6 (QtWebEngine included).
set -euo pipefail
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
PY="${TAPEDECK_TRAY_PY:-/usr/bin/python3}"
"$PY" -c 'import PySide6.QtWebEngineWidgets' 2>/dev/null \
  || { echo "$PY has no PySide6 with QtWebEngine; set TAPEDECK_TRAY_PY"; exit 1; }
APPS="$HOME/.local/share/applications"
mkdir -p "$APPS"
cat > "$APPS/tapedeck.desktop" <<DESK
[Desktop Entry]
Type=Application
Name=∞ TAPEDECK
Comment=Infinite personal radio: tray, deck window, captioning, quit frees the GPU
Exec=$PY $HERE/tapedeck_tray.py --open
Icon=$HERE/tapedeck.svg
Terminal=false
Categories=AudioVideo;Audio;
StartupWMClass=tapedeck
DESK
echo "menu entry: $APPS/tapedeck.desktop"
if [ "${1:-}" = "--autostart" ]; then
  mkdir -p "$HOME/.config/autostart"
  sed 's/ --open$//' "$APPS/tapedeck.desktop" > "$HOME/.config/autostart/tapedeck.desktop"
  echo "autostart: $HOME/.config/autostart/tapedeck.desktop"
fi
