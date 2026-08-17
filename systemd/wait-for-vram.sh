#!/usr/bin/env bash
# Block until the GPU has room for a CUDA context, then exit 0.
#
# ComfyUI cannot start while the AI listening pass holds the card: it dies in
# mem_get_info() before it can even create a context, and systemd then crash
# -loops it — 19 restarts in one measured case, each loading torch, stealing
# CPU from the very dub it is waiting on, with the deck down throughout.
#
# Capping the captioner instead does not work on a 16 GB card: the 8-bit model
# plus its activations need essentially all of it, and a budget small enough to
# leave headroom fails every track on a 300 MB allocation. So the music server
# waits its turn rather than fighting for a card that is genuinely full.
#
# Usage (in the unit, before ExecStart):
#   ExecStartPre=/path/to/tapedeck/systemd/wait-for-vram.sh 1000 0
# Args: required free MB (default 1000), timeout in seconds (0 = wait forever).
set -u
NEED_MB=${1:-1000}
TIMEOUT=${2:-0}
INTERVAL=10
waited=0

# Ask, rather than wait for luck. The listening pass checks this flag at each
# track boundary and releases its allocator cache once when it is set, which
# opens a multi-GB window within one track (~40s) instead of leaving the deck
# down for the rest of the dub. Removed on every exit path, so a pass never
# keeps paying for a server that already started or gave up.
BASE=${TAPEDECK_BASE:-$(cd "$(dirname "$0")/.." && pwd)}
WANT="$BASE/radio/WANT_CARD"
cleanup() { rm -f "$WANT" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

free_mb() {
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null \
        | head -1 | tr -d ' '
}

while :; do
    free=$(free_mb)
    # No nvidia-smi, or an unreadable answer: do not block the boot on a
    # check we cannot make.
    case "$free" in
        ''|*[!0-9]*) echo "wait-for-vram: cannot read free VRAM — starting anyway"
                     exit 0 ;;
    esac
    if [ "$free" -ge "$NEED_MB" ]; then
        [ "$waited" -gt 0 ] && echo "wait-for-vram: ${free} MB free after ${waited}s — starting"
        exit 0
    fi
    if [ "$TIMEOUT" -gt 0 ] && [ "$waited" -ge "$TIMEOUT" ]; then
        echo "wait-for-vram: still only ${free} MB free after ${waited}s — starting anyway"
        exit 0
    fi
    if [ "$waited" = 0 ]; then
        echo "wait-for-vram: only ${free} MB free, need ${NEED_MB} — asking the listening pass to make room"
        mkdir -p "$(dirname "$WANT")" 2>/dev/null || true
        : > "$WANT" 2>/dev/null || true
    fi
    sleep "$INTERVAL"
    waited=$((waited + INTERVAL))
done
