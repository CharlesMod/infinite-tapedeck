#!/usr/bin/env python3
"""Caption the corpus with Music Flamingo (8-bit): one rich prose description
per track — genre, mood arc, instrumentation, production character, vocal
character. Output feeds the Gemma rewrite into Music3's caption schema and
the essence-card enrichment.

GPU coordination: raises the tank daemon's PAUSE flag for the whole pass and
waits for any in-flight generation to drain before loading the model, so the
card is never contested. Resume-safe (path+hash keys), ~10-20s per track.

Usage: venv/bin/python analysis/caption_pass.py [--one] (self-test: first
uncaptioned track only, prints the result)
"""
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

import os as _os
BASE = _os.environ.get("TAPEDECK_BASE") or _os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))


LIB = f"{BASE}/library"
if "--source" in sys.argv:
    LIB = sys.argv[sys.argv.index("--source") + 1]
# yield the GPU back to generation when the radio's tank runs low (seconds
# of unconsumed audio for the ACTIVE station; 0 disables the duty cycle)
YIELD_BELOW_S = 0
if "--yield-below" in sys.argv:
    YIELD_BELOW_S = float(sys.argv[sys.argv.index("--yield-below") + 1])
OUT = f"{BASE}/analysis/captions.jsonl"
if "--out" in sys.argv:
    OUT = sys.argv[sys.argv.index("--out") + 1]
PAUSE_FLAG = f"{BASE}/radio/PAUSE"
# Flamingo wants ~11 GB, so the pass must not start while a generation is in
# flight. Check the configured generator plus the conventional ports — a host
# that is down simply reports nothing, and guessing wrong here silently
# removes the collision guard.
SERVERS = ["http://127.0.0.1:8188", "http://127.0.0.1:8189"]
_cfg = {}
try:
    with open(f"{BASE}/radio/config.json") as _f:
        _cfg = json.load(_f)
    _host = _cfg.get("comfy_host")
    if _host and _host not in SERVERS:
        SERVERS.insert(0, _host)
except (OSError, ValueError):
    pass

# VRAM this pass must leave on the card, always.
#
# Flamingo will happily grow into the last byte of a 16 GB card, and then
# ComfyUI cannot be restarted at all: it dies in mem_get_info() before it can
# even create a CUDA context, and systemd crash-loops it. Measured on a
# 443-track dub — 13.2 GB held, 71 MB free, 19 restarts, deck down until the
# dub finished. The captioner is *designed* to run alongside the music server,
# so it is the captioner's job to stay small enough for that to be true.
VRAM_HEADROOM_MB = int(_cfg.get("caption_vram_headroom_mb", 1400))
MODEL = "nvidia/music-flamingo-2601-hf"
EXTS = (".flac", ".mp3", ".m4a", ".ogg", ".opus", ".wav")
MAX_INPUT_S = 360  # >6 min tracks: middle 6 minutes is the representative cut

PROMPT = (
    "Describe this music in rich, specific prose for a producer who will "
    "recreate its style. Cover: genre and subgenre; tempo feel and estimated "
    "BPM; key/mode feel; the mood and how it evolves across the track; every "
    "prominent instrument and what it does; the arrangement arc "
    "(intro/build/peak/outro); production character (space, warmth, "
    "compression, era); and vocal character if any vocals exist (timbre, "
    "delivery, language) or state it is instrumental. Be concrete. "
    "No artist guesses, no filler."
)


BREADCRUMB = os.path.join(os.path.dirname(OUT), "caption_attempting.txt")


def _stopped(*_):
    """SIGTERM is the pipeline, the deck's cancel button, or a person —
    none of which is the track killing the process. Leaving the breadcrumb
    behind would blame a perfectly good file and skip it forever after."""
    try:
        os.unlink(BREADCRUMB)
    except OSError:
        pass
    try:
        os.unlink(PAUSE_FLAG)
    except OSError:
        pass
    sys.exit(143)


def radio_is_live():
    """Is there a running radio to yield the card TO?

    The duty cycle exists so a long pass does not starve a playing station.
    On a first capture there is no daemon, no essence cards and an empty
    tank — yielding there hands the GPU to nobody and waits forever for a
    tank nothing is filling. The daemon rewrites daemon_state.json on every
    log line, so a fresh timestamp is a real heartbeat."""
    try:
        with open(f"{BASE}/radio/daemon_state.json") as f:
            return time.time() - int(json.load(f).get("ts", 0)) < 300
    except (OSError, ValueError, TypeError, AttributeError):
        return False


def content_hash(path):
    h = hashlib.blake2b(digest_size=16)
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def queue_busy():
    for host in SERVERS:
        try:
            req = urllib.request.urlopen(host + "/queue", timeout=5)
            q = json.load(req)
            if q.get("queue_running") or q.get("queue_pending"):
                return True
        except Exception:
            continue  # server down = not busy
    return False


FLAMINGO_VRAM_MB = 12000  # 8-bit weights plus activations


def _smi(field):
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={field}",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5).stdout.strip()
        return int(out.splitlines()[0])
    except Exception:
        return None


def free_vram_mb():
    return _smi("memory.free")


def total_vram_mb():
    return _smi("memory.total")


CONTEXT_MB = 500  # roughly what ComfyUI needs just to create a CUDA context


def free_the_card():
    """An idle queue is not an idle card — ComfyUI holds model weights
    between jobs, and Flamingo cannot load into what is left. Ask each
    server to drop its weights, then wait for the VRAM to actually come
    back before committing to an 8-bit load."""
    for host in SERVERS:
        try:
            req = urllib.request.Request(
                host + "/free",
                data=json.dumps({"unload_models": True,
                                 "free_memory": True}).encode(),
                headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=15).read()
        except Exception:
            continue  # server down, or an older build without /free
    for _ in range(15):
        free = free_vram_mb()
        if free is None or free >= FLAMINGO_VRAM_MB:
            return free
        time.sleep(2)
    return free_vram_mb()


def main():
    signal.signal(signal.SIGTERM, _stopped)
    signal.signal(signal.SIGINT, _stopped)
    one = "--one" in sys.argv
    done = set()
    if os.path.exists(OUT):
        with open(OUT) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["path"])
                except Exception:
                    pass

    todo = []
    for root, _, files in os.walk(LIB):
        for name in sorted(files):
            if name.lower().endswith(EXTS):
                rel = os.path.relpath(os.path.join(root, name), LIB)
                if rel not in done:
                    todo.append(rel)
    print(f"{len(todo)} tracks to caption ({len(done)} done)", flush=True)
    if not todo:
        print(f"RESULT {len(done)} already described, nothing new",
              flush=True)
        return
    if one:
        todo = todo[:1]

    poison_path = os.path.join(os.path.dirname(OUT), "caption_poison.json")
    breadcrumb = BREADCRUMB
    poison = set()
    if os.path.exists(poison_path):
        with open(poison_path) as f:
            poison = set(json.load(f))
    # a stale breadcrumb means the last run died natively (SIGABRT) while
    # processing that track — no Python handler ran, so record it here
    if os.path.exists(breadcrumb):
        with open(breadcrumb) as f:
            victim = f.read().strip()
        if victim and victim not in done:
            poison.add(victim)
            with open(poison_path, "w") as f:
                json.dump(sorted(poison), f, indent=1)
            print(f"POISON (from breadcrumb, native crash): {victim}", flush=True)
        os.unlink(breadcrumb)
    if poison:
        skipped = len([t for t in todo if t in poison])
        todo = [t for t in todo if t not in poison]
        print(f"skipping {skipped} poison track(s)", flush=True)

    with open(PAUSE_FLAG, "w") as f:
        f.write(str(os.getpid()))  # lets guards distinguish live vs stale
    try:
        print("PAUSE raised; waiting for tank generation to drain", flush=True)
        while queue_busy():
            time.sleep(15)
        free = free_the_card()
        if free is not None:
            print(f"{free} MB VRAM free after unloading the generator",
                  flush=True)
            if free < FLAMINGO_VRAM_MB:
                print(f"NOT ENOUGH VRAM: Music Flamingo needs about "
                      f"{FLAMINGO_VRAM_MB} MB and only {free} MB is free. "
                      "Something else is holding the card (a game, another "
                      "model, a second ComfyUI). Free it and re-run — the "
                      "pass resumes where it stopped.", flush=True)
                sys.exit(1)

        import logging

        import numpy as np
        import librosa
        import torch
        from transformers import (AutoProcessor, BitsAndBytesConfig,
                                  MusicFlamingoForConditionalGeneration)

        # bitsandbytes logs "MatMul8bitLt: inputs will be cast..." once per
        # matmul — nearly 3 million lines and 240 MB of log across one real
        # corpus, which also floods the deck's capture readout. The cast is
        # expected for an 8-bit model; we do not need to hear about it.
        logging.getLogger("bitsandbytes.autograd._functions").setLevel(
            logging.ERROR)

        # NOT capped with max_memory / set_per_process_memory_fraction.
        # Tried and reverted: on a 16 GB card Flamingo's 8-bit weights are
        # ~11.5 GB and its audio-encoder activations another ~1.5 GB, so any
        # cap that leaves real headroom leaves too little for the model — a
        # 12 GB budget loaded the weights fine and then failed EVERY track on
        # a 334 MB activation. The card is simply not big enough to host both
        # this model and a spare CUDA context by squeezing the model.
        # Headroom is made by releasing cache below, and by the music server
        # waiting for the card instead of crash-looping (see systemd/).
        print("loading Music Flamingo 8-bit", flush=True)
        processor = AutoProcessor.from_pretrained(MODEL)
        model = MusicFlamingoForConditionalGeneration.from_pretrained(
            MODEL,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            device_map="auto")
        model.eval()
        sr = processor.feature_extractor.sampling_rate

        t_all = time.time()
        for i, rel in enumerate(todo, 1):
            if YIELD_BELOW_S and radio_is_live():
                sys.path.insert(0, f"{BASE}/radio")
                import stations
                level = stations.tank_level(stations.active())
                if level < YIELD_BELOW_S:
                    print(f"YIELD tank at {level:.0f}s < {YIELD_BELOW_S:.0f}s "
                          "— handing the GPU back to generation", flush=True)
                    sys.exit(4)
            # The allocator keeps freed blocks, which nvidia-smi still counts
            # as used, so the card looks full even when this pass is between
            # tracks. Handing them back is what lets the music server create
            # a CUDA context without this pass giving up the model.
            if i % 3 == 0:
                free_now = free_vram_mb()
                if free_now is not None and free_now < VRAM_HEADROOM_MB:
                    torch.cuda.empty_cache()
                    after = free_vram_mb()
                    print(f"released cache — free VRAM {free_now} MB -> "
                          f"{after} MB", flush=True)
            path = os.path.join(LIB, rel)
            with open(breadcrumb, "w") as f:
                f.write(rel)
            try:
                dur = librosa.get_duration(path=path)
                if dur > MAX_INPUT_S:
                    off = (dur - MAX_INPUT_S) / 2
                    y, _ = librosa.load(path, sr=sr, mono=True,
                                        offset=off, duration=MAX_INPUT_S)
                else:
                    y, _ = librosa.load(path, sr=sr, mono=True)
                # a NaN, inf, or out-of-range sample can trip a device-side
                # assert that poisons the CUDA context for every later track
                y = np.clip(np.nan_to_num(y, nan=0.0, posinf=1.0,
                                          neginf=-1.0), -1.0, 1.0)
                if len(y) < sr * 3:
                    raise ValueError(f"too short: {len(y)/sr:.1f}s")

                conv = [{"role": "user", "content": [
                    {"type": "audio", "audio": y},
                    {"type": "text", "text": PROMPT}]}]
                inputs = processor.apply_chat_template(
                    conv, add_generation_prompt=True, tokenize=True,
                    return_dict=True, return_tensors="pt").to(
                        model.device, torch.bfloat16)  # casts float tensors only
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=512,
                                         do_sample=False)
                text = processor.batch_decode(
                    out[:, inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True)[0].strip()

                with open(OUT, "a") as f:
                    f.write(json.dumps({
                        "path": rel, "hash": content_hash(path),
                        "duration_s": round(dur, 1),
                        "caption": text}) + "\n")
                if one:
                    print("----\n", text, "\n----", flush=True)
                    return
                if i % 10 == 0:
                    rate = (time.time() - t_all) / i
                    print(f"[{i}/{len(todo)}] {rate:.1f}s/track, "
                          f"~{rate*(len(todo)-i)/60:.0f} min left", flush=True)
            except Exception as e:
                print(f"FAIL {rel}: {e!r:.120}", flush=True)
                if "device-side assert" in repr(e) or "AcceleratorError" in repr(e):
                    # context is poisoned: record the trigger, die fast, and
                    # let the next (clean) run skip it and continue
                    poison.add(rel)
                    with open(poison_path, "w") as f:
                        json.dump(sorted(poison), f, indent=1)
                    print(f"POISON {rel} — exiting for a clean context", flush=True)
                    sys.exit(3)
                if one:
                    sys.exit(1)  # a failed self-test must LOOK failed
            try:
                os.unlink(breadcrumb)
            except FileNotFoundError:
                pass
            print(f"PROG {i} {len(todo)} {rel}", flush=True)
        wrote = sum(1 for _ in open(OUT)) if os.path.exists(OUT) else 0
        print(f"RESULT {wrote} track(s) described by ear", flush=True)
    finally:
        try:
            os.unlink(PAUSE_FLAG)
            print("PAUSE lowered", flush=True)
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
