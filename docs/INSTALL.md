# Installing ∞ TAPEDECK

**Read this line before typing anything: tapedeck has no virtualenv of its
own.** The deck is a ComfyUI custom node and the analysis stages are launched
by the same interpreter that runs ComfyUI, so everything — ComfyUI, the
listener stack, the radio daemon — lives in **one** environment.

Every command below refers to that one interpreter as `$COMFY_PY`. Set it
once, in step 1, and the rest of this document is unambiguous.

Order matters: ComfyUI and its environment (1) → model weights (2) → tapedeck
itself (3) → dependencies (4) → check (5) → capture (6) → run (7).

---

## 0. What you need

- **Linux.** Python 3.10+ (3.12 verified), an NVIDIA GPU with **16 GB VRAM**
  (RTX 4080/5080 class)
- ~15 GB disk for the Music 3 weights, plus room for generated audio
- Optional: another ~17 GB if you want the AI listening pass (recommended —
  see step 6)

### On Windows: use WSL2, not native Windows

Native Windows is **not supported**, and it will not work if you try. The
radio identifies who is holding the GPU by reading `/proc/<pid>/cmdline`,
coordinates the captioner and the generator with POSIX signals, and asks
systemd for unit PIDs — none of which exist on Windows. It is not a matter of
path separators.

WSL2 works and is the supported Windows route: it has CUDA passthrough to
your NVIDIA card, a real `/proc`, and systemd (enable it with
`systemd=true` under `[boot]` in `/etc/wsl.conf`). Install Ubuntu under WSL2,
then follow this guide unchanged from inside it. Two notes:

- Keep the repo and your music **inside** the Linux filesystem (`~/tapedeck`,
  not `/mnt/c/...`). Analysis reads every file end to end, and the
  `/mnt/c` bridge is slow enough to dominate the capture.
- The deck is a web page, so open it in Windows' own browser at
  `http://localhost:8188/...` — WSL2 forwards localhost automatically.

macOS is not supported either: MiniMax Music 3 needs CUDA.

---

## 1. ComfyUI 0.33+ and its environment

If you already run ComfyUI, skip the install and just point `COMFY_PY` at the
python it uses.

```bash
git clone https://github.com/comfyanonymous/ComfyUI
cd ComfyUI
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
```

Now fix the interpreter for the rest of this document:

```bash
export COMFY_PY="$PWD/venv/bin/python"   # inside the ComfyUI directory
export COMFY_DIR="$PWD"
$COMFY_PY -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

That last line must print `True`. If it prints `False`, install the torch
build matching your CUDA version from [pytorch.org](https://pytorch.org)
before going on — nothing downstream will work without it.

> Using conda, uv, pyenv or a system python instead? Fine. `COMFY_PY` is
> whatever interpreter can `import torch` **and** runs ComfyUI. There is no
> requirement that it be a venv.

## 2. MiniMax Music 3 weights (~14 GB)

From [Comfy-Org/MiniMax-Music-3](https://huggingface.co/Comfy-Org/MiniMax-Music-3)
into ComfyUI's model folders:

| file | goes in |
|---|---|
| `minimax_music3_dit_fp16.safetensors` | `$COMFY_DIR/models/diffusion_models/` |
| `minimax_music3_text_encoder_pruned_int8_convrot.safetensors` | `$COMFY_DIR/models/text_encoders/` |
| `minimax_music3_dav.safetensors` | `$COMFY_DIR/models/vae/` |

Using a different quantization of the DiT? Set `dit_model` in
`radio/config.json` (step 7) to its filename.

## 3. This repo

Clone it wherever you like — it does **not** have to live inside ComfyUI.

```bash
git clone https://github.com/CharlesMod/infinite-tapedeck tapedeck
cd tapedeck
export TAPEDECK="$PWD"

# make the deck visible to ComfyUI
ln -s "$TAPEDECK/comfyui_node/music_studio" "$COMFY_DIR/custom_nodes/"

# tell the node where the repo is (it reads this file at startup)
echo "$TAPEDECK" > comfyui_node/music_studio/tapedeck_base.txt
```

All state lives under the repo: `library/` (your music), `stations/`,
`analysis/`, `radio/tank/`. To keep state somewhere else, set
`TAPEDECK_BASE=/elsewhere` in the environment of everything you launch.

## 4. Dependencies

Into the **same** interpreter, not a new venv:

```bash
$COMFY_PY -m pip install -r requirements.txt
```

That is librosa, soundfile, av, scikit-learn and transformers, plus
bitsandbytes and accelerate for the AI listening pass. ComfyUI already
supplies torch and numpy.

## 5. Check the install before it does an hour of work

```bash
$COMFY_PY analysis/preflight.py
```

It verifies the interpreter, every import, audio decoding for the formats
actually in your library, VRAM, disk, the three model files, and whether
ComfyUI is reachable. **Fix every `FAIL` line before continuing** — each one
is a failure that would otherwise surface much later, in a much less obvious
form. Warnings are safe to proceed past.

## 6. Capture your library

Put music in `library/`, then:

```bash
$COMFY_PY analysis/import_pipeline.py --station full-library
```

This runs inventory → features → CLAP embeddings → **the AI listening pass**
→ taste veins → essence cards. It is resumable and incremental: re-running
only processes new or changed files.

### The listening pass runs by default, and should

[Music Flamingo](https://huggingface.co/nvidia/music-flamingo-2601-hf)
listens to every track and writes a paragraph describing it — genre, tempo
feel, instruments, arrangement arc, production character — and those words
steer everything the generator does afterwards.

It is on by default because the alternative is bad: without it a station is
described only by numbers (tempo, brightness, energy shape, note density).
Nothing in that says *techno*, or *fingerpicked guitar*, or *analog tape
saturation*. The generator fills the gap with whatever it likes, the result
drifts off-genre, and the critic then rejects most of it.

The cost is about 15 seconds of GPU time per track (≈10 minutes for 40
tracks) plus a one-time 17 GB model download. The pass is resumable, survives
interruption, and yields the GPU back to the radio when the spool runs low.
The capture prints the expected time before it starts.

If the dependencies or the disk space are missing, the capture says so and
continues without captions rather than failing — the radio still runs.

### Skipping it

```bash
$COMFY_PY analysis/import_pipeline.py --station full-library --no-captions
```

If you do skip it, say something in your own words instead:

```bash
$COMFY_PY analysis/import_pipeline.py --station full-library --no-captions \
  --describe "dark melodic techno, analog hardware, hypnotic, no vocals"
```

One sentence, stored with the station and folded into every caption written
from it. Much better than statistics alone, much worse than the listening
pass. You can also drop a `DESCRIPTION.txt` in the station's music folder.
Re-run the capture after either — cards are rebuilt from what is on disk.

## 7. Configure (optional)

`radio/config.json` — every key has a default:

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

- `llm_base` / `llm_model`: any OpenAI-compatible endpoint (llama.cpp,
  llama-swap, Ollama), used for caption variety and lyrics. Set `llm_base`
  to `null` to disable the LLM — the radio then runs on essence-card seeds.
  `llm_model: null` uses the first model the endpoint offers.
- `sibling_hosts`: other ComfyUI instances that outrank the radio on this GPU.
- `critic_keep_frac` (default `0.5`): how picky the critic is when your
  library's own calibrated bar turns out to be higher than generated audio
  can reach. `0.5` keeps the better half of what each vein produces, `0.7` is
  much less picky, `0.0` disables the learned bar entirely and uses only the
  corpus bar. `critic_floor` (default `0.45`) is the absolute similarity
  below which a take is never banked, whatever the learned bar says.
- Also accepted: `min_take_s`, `lyric_cap`, `bundle_queue_target`,
  `tank_target_tracks`, `relief_step`, `relief_max`, `dit_model`.

Single-machine GPU sharing is the default: the daemon frees the generator's
weights, swaps the LLM in, writes `caption_batch` caption/lyric bundles in one
residency, unloads, then generates the batch back to back.

## 8. Run

Start ComfyUI, then the spool daemon:

```bash
$COMFY_PY radio/tank_daemon.py
```

Open **`http://<host>:8188/extensions/music_studio/index.html`** and press
PLAY.

The first tracks take a few minutes: generation is slower than realtime by
design, and the deck waits for takes that pass the critic. Watch the daemon's
log — it narrates every decision.

Systemd templates for ComfyUI, the daemon, and an idle-VRAM watchdog are in
`systemd/`. Edit the paths in each, then `systemctl --user enable --now` them.

---

## Troubleshooting

**`./venv/bin/python: no such file or directory`**
There is no venv inside this repo — that was an error in older instructions.
Use `$COMFY_PY` (step 1), the interpreter that runs ComfyUI.

**`ModuleNotFoundError: No module named 'torch'` (or librosa, sklearn…)**
You are running a different interpreter from the one the packages are in.
Check with `$COMFY_PY analysis/preflight.py`; it prints the interpreter path
it is checking. Install into that same one (step 4). Creating a second venv
for this repo will produce exactly this error.

**`ValueError: need at least one array to stack`**
Older versions surfaced "every track failed to embed" as this numpy error
three stages downstream. Current versions stop at the failing stage and print
the first real failure. Re-run `analysis/preflight.py`: it is almost always a
missing decoder for your audio format, or torch/transformers living in a
different interpreter.

**`scikit-learn is required to cluster N tracks into veins`**
`$COMFY_PY -m pip install scikit-learn`. Libraries under 24 tracks skip
clustering and never need it.

**The capture finds no music**
The capture stops at the inventory stage and prints the folder it searched.
Recognized: flac, mp3, m4a, ogg, opus, wav, wma, aac, aiff. m4a/aac/wma need
ffmpeg on PATH; the rest are handled by libsndfile.

**The radio rejects everything / the spool never fills**
Look for `REJECT … [corpus 0.9xx, …]` in the daemon log. A tight library (one
artist, one album) is so self-similar that its own calibrated bar is higher
than generated audio can reach. The critic detects this: after four scored
takes it switches to a bar learned from what the vein actually produces and
logs `corpus 0.9xx out of reach`. If it is still too strict, raise
`critic_keep_frac` in `radio/config.json` (0.5 = keep the better half; 0.7 =
much less picky).

**Generated tracks do not sound like my library**
Two things to check. Did you run the listening pass or give a `--describe`
line (step 6)? Without one, nothing tells the generator what genre it is
aiming at. And note the honest limit: MiniMax Music 3 is conditioned on
*text*, not audio — tapedeck steers with words and filters with audio
embeddings, so it converges on the style region of your library rather than
cloning its sound.

**Takes come out very short and get rejected as stubs**
Length follows arrangement richness in the caption, not the requested
duration. This is what the LLM endpoint (`llm_base`) is for: it writes
multi-section arrangements. Without one, the radio falls back to card seeds,
which are shorter and plainer.
