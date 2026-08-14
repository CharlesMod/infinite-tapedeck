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

![cyberpunk cassette deck UI](docs/screenshot.png)

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
- **The tank** — generation is slower than realtime, so a background daemon
  pre-fills a buffer of approved takes whenever your GPU is idle, and yields
  instantly when you need the card (games, other AI workloads, a sibling
  ComfyUI instance).
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
- Optional: ~17 GB disk for Music Flamingo if you want AI captions

See **[docs/INSTALL.md](docs/INSTALL.md)** for the full walkthrough.

## Quickstart

```bash
git clone <this repo> tapedeck && cd tapedeck
# 1. install ComfyUI 0.33+ and Music 3 weights (docs/INSTALL.md)
# 2. link the deck into ComfyUI
ln -s "$PWD/comfyui_node/music_studio" /path/to/ComfyUI/custom_nodes/
echo "$PWD" > comfyui_node/music_studio/tapedeck_base.txt
# 3. put music in ./library (or point a station anywhere later)
# 4. capture it
python analysis/import_pipeline.py --station full-library
# 5. run the tank daemon + open the deck
python radio/tank_daemon.py &
# http://<host>:8188/extensions/music_studio/index.html — press PLAY
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
