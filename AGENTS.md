# AGENTS.md — helping someone set up and run ∞ TAPEDECK

You are an AI assistant helping a user install, run or fix TAPEDECK: a local
AI radio that learns a music folder and generates endless new songs like it
(ComfyUI + MiniMax Music 3, CLAP critic, optional NVIDIA Music Flamingo
captions). Read this file first; it is the shortest path to a working deck.

## The 30-second version

```bash
git clone https://github.com/CharlesMod/infinite-tapedeck tapedeck && cd tapedeck
./install.sh --music /path/to/their/music
```

That is the whole install. It is idempotent: if anything fails, fix the cause
and run the same command again — finished steps are skipped. When it ends it
prints the deck URL. Tell the user to:

1. open the URL and press **PLAY** — the deck plays their own music at once,
   generated takes join as they are made;
2. press **Accept & start listening pass** (Dubbing section, right panel) to
   allow the one-time ~17 GB NVIDIA Music Flamingo download. This is the
   user's decision — do not accept it for them unless they explicitly said
   yes (then `--accept-flamingo` on a capture, or POST
   `/music_studio/consent {"accept": true}`, records it).

## Windows

Use `install.ps1` (PowerShell). It installs WSL2 if needed (admin, then one
reboot, then run it again), installs Ubuntu-24.04 (the user creates a Linux
username once), checks the GPU inside WSL, copies the music into the Linux
filesystem (`~/tapedeck-music`; `-LinkMusic` reads it from `/mnt/c` instead,
slower), runs `install.sh` inside WSL, and creates Start menu/desktop
shortcuts (`%LOCALAPPDATA%\Tapedeck\tapedeck-start.cmd`). WSL stops a distro
soon after its last console closes, so the TAPEDECK window must stay open
(minimised) while the radio runs. Inside WSL everything below applies
unchanged; reach it with `wsl -d Ubuntu-24.04`. Native Windows (no WSL) is
not supported: the radio uses /proc, POSIX signals and systemd.

If the GPU is not visible inside WSL: update the NVIDIA Windows driver (do
NOT install a Linux NVIDIA driver inside WSL), then `wsl --update`.

## macOS

Not supported yet. The generator (MiniMax Music 3 in ComfyUI) needs CUDA;
whether it runs on Apple's MPS backend is untested, and would need roughly
32 GB+ unified memory. The listening pass loads Flamingo in 8-bit through
bitsandbytes, which is CUDA-only. Porting order if someone tries: (1) load
Music 3 in ComfyUI on MPS and time one take; only if that works, (2) replace
nvidia-smi/`/proc` GPU-sharing code in radio/tank_daemon.py and
analysis/caption_pass.py, (3) load Flamingo in fp16 on MPS.

## Before you run anything, check

| check | command | needs |
|---|---|---|
| OS | `uname -s` | Linux, or Windows **inside WSL2** (native Windows/macOS unsupported) |
| GPU | `nvidia-smi --query-gpu=name,memory.total --format=csv` | NVIDIA, ~16 GB VRAM |
| disk | `df -h <install dir>` | ~45 GB free (env 8 + weights 14 + Flamingo 17 + audio) |
| music | `find <dir> -type f -iname '*.flac' -o -iname '*.mp3' \| wc -l` | a local folder of audio files |

**Music on a NAS / SMB / NFS share:** copy it to local disk first (audio
files only) and point `--music` at the copy. Analysis reads every file end to
end many times; over a network mount that dominates the capture and a dropped
mount fails it. `rclone copy <remote>:music ./music --include '*.{flac,mp3,ogg,m4a,opus,wav}'`
works without root.

## Installer options

`./install.sh --help` lists them. The useful ones:

- `--music DIR` — the music folder; `library/` becomes a symlink to it
  (never copied or modified).
- `--weights-from DIR` — reuse existing MiniMax Music 3 files (searched by
  filename) instead of downloading 14 GB.
- `--comfy DIR` / `--python PATH` — use an existing ComfyUI and its
  interpreter. Tapedeck must run in **ComfyUI's** interpreter; there is no
  separate tapedeck venv (a `venv/` in the repo would be picked up by the
  capture pipeline and break it).
- `--port N` — when 8188 is taken (writes `comfy_host` to `radio/config.json`).
- `--cuda cu126|cu128` — PyTorch wheel; RTX 50xx (Blackwell) needs cu128+.
- `--no-start`, `--no-capture`, `-y`.

## Layout after install

```
tapedeck/
  install.sh           one-command installer
  bin/tapedeck         control command (linked into ~/.local/bin)
  run/                 tapedeck.env (paths), *.pid, comfy.log, daemon.log
  ComfyUI/             ComfyUI + venv/ (or a symlink to --comfy). The daemon
                       reads finished takes from ComfyUI/output — keep it here
  library/             the music (symlink to --music)
  analysis/            capture pipeline + its outputs (features.jsonl, ...)
  radio/               tank_daemon.py (generation), stations, spool, settings
  comfyui_node/music_studio/   the deck: server routes (__init__.py) + web/
```

## Running it

```bash
tapedeck status     # comfy/daemon running?, capture stage and %, deck URL
tapedeck start | stop | restart
tapedeck logs       # follows run/comfy.log and run/daemon.log
tapedeck capture    # re-capture the active station after adding music
```

`tapedeck tray` (tray/tapedeck_tray.py) owns the same run/*.pid processes:
radio on/off writes/removes radio/DISABLED (tank_daemon refuses to start while
it exists), captioning pause/resume calls /music_studio/import/{pause,resume},
Quit pauses captioning, stops the daemon, POSTs /interrupt + /free, then stops
ComfyUI.

Capture progress: `analysis/import_progress.json` (stage, track_i/track_n,
overall_pct, `captions_pending_consent`), log in `analysis/capture.log`.
Stages: inventory → features (CPU, parallel) → embeddings (GPU) → captions
(GPU, needs consent) → veins → cards. Everything is cached per track: a
re-run only processes new or changed files.

## Things that look broken but are not

- **GPU near 0 % during capture**: the features stage is CPU work (librosa).
  The GPU stages come after it.
- **"station has no capture yet — holding"** in the daemon log: generation
  waits for the first capture to finish. The deck plays library tracks
  meanwhile.
- **Listening pass skipped**: consent not given yet — the deck shows the
  Accept button; `tapedeck status` says so too.

## Common failures

| symptom | cause / fix |
|---|---|
| `torch cannot see the GPU` | driver too old for the wheel — rerun with `--cuda cu126`; check `nvidia-smi` works |
| ComfyUI exits at start | `tail run/comfy.log`; usually a port in use (`--port`) or a missing model file |
| preflight `FAIL no audio files` | `library/` empty or a broken symlink; rerun with `--music DIR` |
| some tracks fail to decode | install `ffmpeg` (system package) |
| takes generate but never reach the deck | `ComfyUI/output` is not where the daemon looks — `ComfyUI` must be in (or symlinked into) the repo |
| capture slow on features | it uses all cores but one (`TAPEDECK_RESERVE_CPU`, `TAPEDECK_FEATURE_WORKERS`) |
| CUDA out of memory | another process holds the card (game, other model); the radio yields and retries |

## Rules for you, the assistant

- Stop processes by the PIDs in `run/*.pid` (or `tapedeck stop`). Never
  `pkill -f` a pattern — it can match your own shell.
- Do not commit runtime state: `run/`, `ComfyUI`, `library/*`, `radio/*.json`
  settings, `analysis/*.json*` are git-ignored for a reason.
- Never accept the Flamingo download on the user's behalf without their yes.
- Changing the deck UI: `comfyui_node/music_studio/web/index.html` is served
  statically (reload the page); server changes in `__init__.py` need
  `tapedeck restart`.
- Verify a layout change by measuring, not guessing: headless Chromium via
  Playwright works for screenshots, but some builds crash on form controls —
  stub `<input>/<select>/<audio>` in a routed copy of the page when measuring.
