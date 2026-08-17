<img width="1333" height="1045" alt="image-1786745043593" src="https://github.com/user-attachments/assets/bb15f061-fc59-495e-b50b-9ecc93b2e510" />
# ∞ TAPEDECK — Infinite Personal AI Radio, Fully Local

**Turn a folder of songs into a radio station that never ends.** TAPEDECK is a
self-hosted, open-source AI music generation radio built on
[ComfyUI](https://github.com/comfyanonymous/ComfyUI) and
[MiniMax Music 3](https://huggingface.co/MiniMaxAI/MiniMax-Music3) (open
weights). It listens to your music library, learns your taste, and generates
an endless stream of new songs *like* yours — but new — on your own GPU. No
cloud, no subscription, no telemetry. Just you, one graphics card, and a
cassette deck from a cyberpunk 1994.

> Drop a folder of hand-picked songs → get a **Pandora-style AI radio
> station** for that exact mood, running infinitely, locally.

## What it does

- **Local AI music generation radio** — full songs (vocals or instrumental)
  generated with MiniMax Music 3 open weights on a single consumer GPU
  (16 GB VRAM, tested on an RTX 5080).
- **Learns your taste from your own library** — a listener stack (librosa
  features + [CLAP](https://huggingface.co/laion/clap-htsat-unfused) audio
  embeddings + optional
  [NVIDIA Music Flamingo](https://huggingface.co/nvidia/music-flamingo-2601-hf)
  captions) distills your collection into taste *veins* and essence cards.
- **Stations = folders.** Your whole library is a station. Twelve songs you
  love at 2 a.m. is a station. Under 24 tracks, no clustering — the folder
  *is* the mood.
- **A critic with your ears** — every generated take is CLAP-scored against
  your corpus before you hear it; off-taste takes are rejected automatically,
  with thresholds self-calibrated per vein from your own music.
- **The spool** — generation is slower than realtime, so a background daemon
  keeps a reserve of approved takes wound ahead whenever your GPU is idle, and
  yields instantly when you need the card (games, other AI workloads, a
  sibling ComfyUI instance).
- **Feedback that steers** — keep / skip / dislike buttons re-weight what
  gets generated next. The radio converges on you.
- **90s cassette deck UI** — spinning reels, VU meters driven by the actual
  audio signal, VFD phosphor readouts, tape counter. Works from any browser
  on your LAN, phone included.
- **Drag-and-drop library management** — drop songs or folders onto the deck,
  rescan, exclude tracks from a station's essence non-destructively.

## How it works

```
your music folder
      │  capture (one command / one click)
      ▼
features (BPM, key, energy arc) ──┐
CLAP embeddings ──────────────────┼──► taste veins ──► essence cards
optional AI captions (Flamingo) ──┘         │
                                            ▼
                    caption sampler (+ optional local LLM for lyrics)
                                            │
                                            ▼
              MiniMax Music 3 generation (ComfyUI, your GPU)
                                            │
                                            ▼
                    CLAP critic — sounds like your taste? no → reject
                                            │ yes
                                            ▼
                              the spool → your radio deck
```

Generation and analysis share one GPU through a duty cycle: captioning yields
to the radio when the spool runs low and resumes once it rewinds.

## Requirements

- Linux, Python 3.12, an NVIDIA GPU with **16 GB VRAM** (RTX 4080/5080 class)
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) **0.33+** with the
  MiniMax Music 3 model files (~14 GB, see
  [Comfy-Org/MiniMax-Music-3](https://huggingface.co/Comfy-Org/MiniMax-Music-3))
- Optional: any OpenAI-compatible local LLM endpoint (llama.cpp, llama-swap,
  Ollama) for caption variety and lyric writing — without one, the radio
  still runs on essence-card seeds
- ~17 GB disk for [Music Flamingo](https://huggingface.co/nvidia/music-flamingo-2601-hf),
  which listens to your library and writes what it hears. This runs **by
  default** (`--no-captions` to skip) because without it a station is
  described to the generator only by numbers (tempo, brightness, energy) —
  no genre, no instruments — and takes drift off-style. About 15 s per track,
  resumable, yields the GPU to the radio.

See **[docs/INSTALL.md](docs/INSTALL.md)** for the full walkthrough.

## Quickstart

TAPEDECK has **no virtualenv of its own** — it is a ComfyUI custom node, and
everything runs in the interpreter that runs ComfyUI. Below, that interpreter
is `$COMFY_PY`. Full walkthrough in **[docs/INSTALL.md](docs/INSTALL.md)**.

```bash
# 0. ComfyUI 0.33+ installed, Music 3 weights in place (docs/INSTALL.md 1-2)
export COMFY_PY=/path/to/ComfyUI/venv/bin/python
export COMFY_DIR=/path/to/ComfyUI

# 1. clone anywhere, link the deck into ComfyUI
git clone https://github.com/CharlesMod/infinite-tapedeck tapedeck && cd tapedeck
ln -s "$PWD/comfyui_node/music_studio" "$COMFY_DIR/custom_nodes/"
echo "$PWD" > comfyui_node/music_studio/tapedeck_base.txt

# 2. dependencies into that SAME interpreter
$COMFY_PY -m pip install -r requirements.txt

# 3. check the install before it does an hour of work
$COMFY_PY analysis/preflight.py

# 4. put music in ./library, then capture it
#    the AI listening pass runs by default (--no-captions to skip)
$COMFY_PY analysis/import_pipeline.py --station full-library

# 5. run the spool daemon, open the deck, press PLAY
$COMFY_PY radio/tank_daemon.py &
# http://<host>:8188/extensions/music_studio/index.html
```

## FAQ

**Is this another AI music generator UI?** No — generators make a song when
you ask. TAPEDECK is a *radio*: it models your taste from music you already
love, generates continuously in the background, rejects its own misses, and
plays an infinite station you never have to prompt.

**Does it copy my songs?** No. Your library is analyzed (tempo, key, energy,
embeddings, optional text descriptions) to steer generation toward a *style
region* — captions never name artists, and the critic compares statistical
audio embeddings, not waveforms.

**How close does it actually get?** It converges on a style region, not a
sound-alike. MiniMax Music 3 is conditioned on *text*: TAPEDECK's listener
stack turns your library into words to steer generation, then uses audio
embeddings to reject takes that miss. Nothing in the open-weights world today
accepts your audio as a conditioning signal, so "in the neighborhood, endlessly"
is the honest ceiling. Run the AI listening pass — the difference between a
station described in prose and one described in BPM numbers is large.

**It rejects everything / the spool never fills.** Usually a very tight
library (one artist, one album): its own calibrated bar sits higher than
generated audio can reach. The critic notices after four takes and switches
to a bar learned from what that vein actually produces — the daemon log says
`corpus 0.9xx out of reach`. Raise `critic_keep_frac` in `radio/config.json`
to make it less picky still.

**Why local?** Your library is personal. Your taste model doubly so. And a
16 GB GPU that games at night can be a radio station by day.

**Keywords**: AI music generation, local AI radio, MiniMax Music 3, ComfyUI
custom node, open weights music model, personal radio, Pandora alternative,
infinite playlist, CLAP embeddings, Music Flamingo, music captioning,
self-hosted, offline music AI, text-to-music, taste model, RTX 5080.

## Credits

Built on the shoulders of: [MiniMax Music 3](https://huggingface.co/MiniMaxAI/MiniMax-Music3)
(music generation, open weights) · [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
(inference server) · [LAION CLAP](https://huggingface.co/laion/clap-htsat-unfused)
(audio-text embeddings) · [NVIDIA Music Flamingo](https://huggingface.co/nvidia/music-flamingo-2601-hf)
(music understanding) · [librosa](https://librosa.org/).

Respect the licenses of the models you download; Music 3 ships under the
MiniMax Community License, Music Flamingo is NVIDIA-noncommercial.

## Status

Early release — extracted from a working single-machine install. The pieces
run in production daily (the author's living room); the *packaged* install
path is young. Issues and PRs welcome.
