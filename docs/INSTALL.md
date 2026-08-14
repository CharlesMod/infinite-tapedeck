# Installing ∞ TAPEDECK

One machine, one GPU, five parts: ComfyUI, the Music 3 weights, this repo,
a capture, and the daemon.

## 1. ComfyUI 0.33+ and a venv

```bash
git clone https://github.com/comfyanonymous/ComfyUI
python3 -m venv venv && ./venv/bin/pip install -r ComfyUI/requirements.txt
./venv/bin/pip install librosa soundfile av transformers
```

## 2. MiniMax Music 3 weights (~14 GB)

From [Comfy-Org/MiniMax-Music-3](https://huggingface.co/Comfy-Org/MiniMax-Music-3)
into ComfyUI's model folders:

| file | goes in |
|---|---|
| `minimax_music3_dit_fp16.safetensors` | `models/diffusion_models/` |
| `minimax_music3_text_encoder_pruned_int8_convrot.safetensors` | `models/text_encoders/` |
| `minimax_music3_dav.safetensors` | `models/vae/` |

## 3. This repo

```bash
git clone <this repo> tapedeck && cd tapedeck
ln -s "$PWD/comfyui_node/music_studio" /path/to/ComfyUI/custom_nodes/
echo "$PWD" > comfyui_node/music_studio/tapedeck_base.txt
```

All state lives under the repo dir: `library/` (your music), `stations/`,
`analysis/`, `radio/tank/`. Set `TAPEDECK_BASE=/elsewhere` to move it.

## 4. Configure (optional)

`radio/config.json` — everything has defaults:

```json
{
  "comfy_host": "http://127.0.0.1:8188",
  "sibling_hosts": [],
  "llm_base": "http://127.0.0.1:8080",
  "llm_model": null,
  "steps": 30,
  "tank_target_s": 10800,
  "caption_batch": 5
}
```

Single-machine GPU sharing is the default: the daemon frees the generator's
weights, swaps the LLM in, writes `caption_batch` caption/lyric bundles in
one residency, unloads (llama-swap's `/unload` is called if present), then
generates the batch back-to-back. `llm_model: null` uses the first model
your endpoint offers; `llm_base: null` disables the LLM entirely.

- `llm_base` / `llm_model`: any OpenAI-compatible endpoint, used for caption
  variety and lyrics. Omit both and the radio runs on essence-card seeds.
- `sibling_hosts`: other ComfyUI instances that outrank the radio on this GPU.
All thresholds are optional config too: `min_take_s`, `lyric_cap`, `caption_batch`, `bundle_queue_target`, `tank_target_tracks`, `relief_step`, `relief_max`.
- AI captions (optional, big): downloads
  [Music Flamingo](https://huggingface.co/nvidia/music-flamingo-2601-hf) on
  first use; needs `bitsandbytes accelerate` installed.

## 5. Capture your library

Put music in `library/` (or point stations at folders later from the UI).

```bash
./venv/bin/python analysis/import_pipeline.py --station full-library
# add --with-captions for the AI listening pass (slow; resumable; yields to the radio)
```

## 6. Run

Start ComfyUI, then:

```bash
./venv/bin/python radio/tank_daemon.py
```

Systemd templates for both (plus an idle-VRAM watchdog) are in `systemd/` —
edit the paths, then `systemctl --user enable --now` each.

Open **`http://<host>:8188/extensions/music_studio/index.html`**. Press PLAY.
The first tracks appear once the deck winds its first accepted takes
(minutes, not seconds — generation is slower than realtime by design).
