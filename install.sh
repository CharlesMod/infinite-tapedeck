#!/usr/bin/env bash
# ∞ TAPEDECK — one-command install (Linux, or Windows via WSL2).
#
#   ./install.sh --music /path/to/your/music
#
# or, without a clone yet:
#
#   curl -fsSL https://raw.githubusercontent.com/CharlesMod/infinite-tapedeck/main/install.sh | bash -s -- --music ~/Music
#
# What it does (each step is skipped if already done, so re-running is safe):
#   1. checks the machine: NVIDIA GPU + VRAM, disk, git, ffmpeg
#   2. gets uv (a fast Python installer) if it is missing — user-local, no sudo
#   3. ComfyUI in <repo>/ComfyUI with Python 3.12, CUDA torch and every
#      dependency, all in ONE environment (tapedeck is a ComfyUI custom node)
#   4. the three MiniMax Music 3 weight files (~14 GB) from Hugging Face,
#      or links to copies you already have (--weights-from)
#   5. links the deck into ComfyUI and points the library at your music
#   6. runs the preflight check, installs the `tapedeck` command
#   7. starts ComfyUI + the radio, starts the first capture, prints the URL
#
# The AI listening pass (NVIDIA Music Flamingo, ~17 GB) is ON by default but
# is never downloaded until you press "Accept & start listening pass" in the
# deck. Until then the capture runs on tempo/key numbers and the deck plays
# your own music while the first takes are made.
#
# Options:
#   --music DIR          your music folder (it is linked, never copied/modified)
#   --comfy DIR          use/create ComfyUI here instead of <repo>/ComfyUI
#   --python PATH        use an existing ComfyUI's python (skip env creation)
#   --weights-from DIR   look for existing MiniMax Music 3 files here first
#   --port N             ComfyUI port (default 8188)
#   --cuda TAG           PyTorch CUDA wheel tag (default cu128; RTX 50xx needs >= cu128)
#   --dir DIR            where to clone tapedeck when run via curl (default ~/tapedeck)
#   --no-start           install only; start later with: tapedeck start
#   --no-capture         start, but do not start the first capture
#   -y                   do not pause on warnings
set -euo pipefail

MUSIC="" COMFY="" PYBIN="" WEIGHTS_FROM="" PORT=8188 CUDA=cu128
CLONE_DIR="$HOME/tapedeck" START=1 CAPTURE=1 YES=0
while [ $# -gt 0 ]; do
  case "$1" in
    --music) MUSIC="$2"; shift 2 ;;
    --comfy) COMFY="$2"; shift 2 ;;
    --python) PYBIN="$2"; shift 2 ;;
    --weights-from) WEIGHTS_FROM="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --cuda) CUDA="$2"; shift 2 ;;
    --dir) CLONE_DIR="$2"; shift 2 ;;
    --no-start) START=0; shift ;;
    --no-capture) CAPTURE=0; shift ;;
    -y|--yes) YES=1; shift ;;
    -h|--help) sed -n '2,40p' "$0" 2>/dev/null || true; exit 0 ;;
    *) echo "unknown option: $1 (see --help)"; exit 2 ;;
  esac
done

c_ok=$'\e[32m' c_warn=$'\e[33m' c_err=$'\e[31m' c_dim=$'\e[2m' c_off=$'\e[0m'
step() { printf '\n%s▶ %s%s\n' "$c_ok" "$*" "$c_off"; }
ok()   { printf '  %s✓%s %s\n' "$c_ok" "$c_off" "$*"; }
warn() { printf '  %s! %s%s\n' "$c_warn" "$*" "$c_off"; WARNED=1; }
die()  { printf '\n%s✗ %s%s\n' "$c_err" "$*" "$c_off"; exit 1; }
WARNED=0

# ---------------------------------------------------------------- repo
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd || echo .)"
if [ -f "$SELF_DIR/comfyui_node/music_studio/__init__.py" ]; then
  TAPEDECK="$SELF_DIR"
else
  step "getting tapedeck"
  command -v git >/dev/null || die "git is required"
  if [ -d "$CLONE_DIR/.git" ]; then git -C "$CLONE_DIR" pull --ff-only -q || true
  else git clone -q https://github.com/CharlesMod/infinite-tapedeck "$CLONE_DIR"; fi
  TAPEDECK="$(cd "$CLONE_DIR" && pwd)"
  ok "repo at $TAPEDECK"
fi
COMFY="${COMFY:-$TAPEDECK/ComfyUI}"
mkdir -p "$TAPEDECK/run"

# ---------------------------------------------------------------- 1. checks
step "checking this machine"
[ "$(uname -s)" = "Linux" ] || die "Linux only (on Windows use WSL2 — see docs/INSTALL.md)"
command -v git >/dev/null || die "git is required"
if command -v nvidia-smi >/dev/null && nvidia-smi >/dev/null 2>&1; then
  read -r GPU_NAME GPU_MB < <(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits \
                              | head -1 | awk -F', ' '{gsub(/ /,"_",$1); print $1, $2}')
  ok "GPU ${GPU_NAME//_/ } (${GPU_MB} MiB)"
  [ "${GPU_MB:-0}" -ge 15000 ] || warn "MiniMax Music 3 wants a 16 GB card; ${GPU_MB} MiB will likely run out of memory"
else
  die "no NVIDIA GPU found (nvidia-smi failed) — the generator needs CUDA"
fi
FREE_GB=$(df -Pk "$TAPEDECK" | awk 'NR==2{print int($4/1048576)}')
if [ "$FREE_GB" -lt 45 ]; then
  warn "only ${FREE_GB} GB free here: ~8 GB environment + ~14 GB weights + ~17 GB listening model + generated audio"
else ok "${FREE_GB} GB free disk"; fi
command -v ffmpeg >/dev/null && ok "ffmpeg" || warn "ffmpeg not found: install it if any of your music fails to decode (preflight checks)"
if [ "$WARNED" = 1 ] && [ "$YES" = 0 ] && [ -t 0 ]; then
  read -r -p "  continue anyway? [Y/n] " a; case "$a" in n|N) exit 1 ;; esac
fi

# ---------------------------------------------------------------- 2. uv
if [ -z "$PYBIN" ]; then
  step "Python tooling (uv)"
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
  if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
    export PATH="$HOME/.local/bin:$PATH"
  fi
  command -v uv >/dev/null || die "could not install uv"
  ok "uv $(uv --version | awk '{print $2}')"
fi

# ---------------------------------------------------------------- 3. ComfyUI + env
step "ComfyUI and its Python environment"
if [ ! -f "$COMFY/main.py" ]; then
  git clone -q --depth 1 https://github.com/comfyanonymous/ComfyUI "$COMFY"
  ok "cloned ComfyUI into $COMFY"
else ok "ComfyUI at $COMFY"; fi
if [ -z "$PYBIN" ]; then
  [ -x "$COMFY/venv/bin/python" ] || uv venv -q --python 3.12 "$COMFY/venv"
  PYBIN="$COMFY/venv/bin/python"
  if ! "$PYBIN" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "  ${c_dim}installing CUDA PyTorch ($CUDA) — a few GB, the slow part${c_off}"
    uv pip install -q --python "$PYBIN" torch torchvision torchaudio \
      --index-url "https://download.pytorch.org/whl/$CUDA"
  fi
  uv pip install -q --python "$PYBIN" -r "$COMFY/requirements.txt"
  uv pip install -q --python "$PYBIN" -r "$TAPEDECK/requirements.txt"
else
  [ -x "$PYBIN" ] || die "--python $PYBIN is not executable"
  "$PYBIN" -m pip install -q -r "$TAPEDECK/requirements.txt"
fi
"$PYBIN" -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('  torch', torch.__version__, '·', torch.cuda.get_device_name(0))" \
  || die "torch cannot see the GPU — try a different --cuda tag (cu126, cu128) to match your driver"
ok "environment ready: $PYBIN"

# ---------------------------------------------------------------- 4. weights
step "MiniMax Music 3 weights (~14 GB)"
WEIGHTS=(diffusion_models/minimax_music3_dit_fp16.safetensors
         text_encoders/minimax_music3_text_encoder_pruned_int8_convrot.safetensors
         vae/minimax_music3_dav.safetensors)
for w in "${WEIGHTS[@]}"; do
  dest="$COMFY/models/$w"; mkdir -p "$(dirname "$dest")"
  if [ -s "$dest" ]; then ok "$(basename "$w") present"; continue; fi
  if [ -n "$WEIGHTS_FROM" ]; then
    found=$(find -L "$WEIGHTS_FROM" -name "$(basename "$w")" -type f 2>/dev/null | head -1)
    if [ -n "$found" ]; then ln -sf "$found" "$dest"; ok "$(basename "$w") linked from $found"; continue; fi
  fi
  echo "  ${c_dim}downloading $w${c_off}"
  "$PYBIN" -c "
from huggingface_hub import hf_hub_download
hf_hub_download('Comfy-Org/MiniMax-Music-3', '$w', local_dir='$COMFY/models')" \
    || die "download of $w failed (re-run to resume)"
  ok "$(basename "$w") downloaded"
done

# ---------------------------------------------------------------- 5. link + music
step "linking the deck into ComfyUI"
ln -sfn "$TAPEDECK/comfyui_node/music_studio" "$COMFY/custom_nodes/music_studio"
echo "$TAPEDECK" > "$TAPEDECK/comfyui_node/music_studio/tapedeck_base.txt"
# the radio reads finished takes from <repo>/ComfyUI/output
[ "$COMFY" = "$TAPEDECK/ComfyUI" ] || ln -sfn "$COMFY" "$TAPEDECK/ComfyUI"
if [ "$PORT" != 8188 ]; then
  "$PYBIN" - "$TAPEDECK/radio/config.json" "$PORT" <<'PY'
import json, os, sys
p, port = sys.argv[1], sys.argv[2]
d = {}
if os.path.exists(p):
    d = json.load(open(p))
d["comfy_host"] = f"http://127.0.0.1:{port}"
json.dump(d, open(p, "w"), indent=2)
PY
fi
ok "deck linked; base $TAPEDECK"

if [ -n "$MUSIC" ]; then
  MUSIC="$(cd "$MUSIC" 2>/dev/null && pwd)" || die "--music: not a directory"
  LIB="$TAPEDECK/library"
  if [ -L "$LIB" ]; then ln -sfn "$MUSIC" "$LIB"
  elif [ -z "$(find "$LIB" -mindepth 1 -not -name .gitkeep 2>/dev/null | head -1)" ]; then
    rm -rf "$LIB"; ln -s "$MUSIC" "$LIB"
  else warn "library/ already has files — left as is (move them out to link $MUSIC instead)"; fi
  ok "library -> $MUSIC"
fi
NTRACKS=$(find -L "$TAPEDECK/library" -type f \( -iname '*.flac' -o -iname '*.mp3' -o -iname '*.ogg' \
          -o -iname '*.m4a' -o -iname '*.opus' -o -iname '*.wav' -o -iname '*.aac' -o -iname '*.aiff' -o -iname '*.wma' \) 2>/dev/null | wc -l)
[ "$NTRACKS" -gt 0 ] && ok "$NTRACKS tracks in the library" \
  || warn "no music yet: put songs in $TAPEDECK/library (or re-run with --music DIR)"

# ---------------------------------------------------------------- 6. control + preflight
step "installing the tapedeck command"
cat > "$TAPEDECK/run/tapedeck.env" <<EOF
TAPEDECK=$TAPEDECK
COMFY=$COMFY
PYBIN=$PYBIN
PORT=$PORT
EOF
mkdir -p "$HOME/.local/bin"
ln -sf "$TAPEDECK/bin/tapedeck" "$HOME/.local/bin/tapedeck"
chmod +x "$TAPEDECK/bin/tapedeck"
ok "tapedeck start | stop | status | logs | capture | open  (in ~/.local/bin)"
case ":$PATH:" in *":$HOME/.local/bin:"*) ;; *) warn "add ~/.local/bin to your PATH to use the tapedeck command";; esac

step "preflight"
COMFYUI_DIR="$COMFY" "$PYBIN" "$TAPEDECK/analysis/preflight.py" || warn "preflight reported problems (above)"

# ---------------------------------------------------------------- 7. start
if [ "$START" = 1 ]; then
  step "starting"
  "$TAPEDECK/bin/tapedeck" start
  if [ "$CAPTURE" = 1 ] && [ "$NTRACKS" -gt 0 ]; then "$TAPEDECK/bin/tapedeck" capture; fi
fi

HOST=$(hostname -s 2>/dev/null || echo localhost)
printf '\n%s∞ TAPEDECK is installed.%s\n' "$c_ok" "$c_off"
echo "  deck:   http://$HOST:$PORT/extensions/music_studio/index.html"
echo "  It plays your own music while the first takes are made."
echo "  Press \"Accept & start listening pass\" in the deck (Dubbing section) to"
echo "  allow the one-time 17 GB NVIDIA Music Flamingo download — recommended."
[ "$START" = 1 ] || echo "  start it with: tapedeck start"
