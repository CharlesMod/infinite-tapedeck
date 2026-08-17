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
try:
    with open(f"{BASE}/radio/config.json") as _f:
        _host = json.load(_f).get("comfy_host")
    if _host and _host not in SERVERS:
        SERVERS.insert(0, _host)
except (OSError, ValueError):
    pass
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


def main():
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
        return
    if one:
        todo = todo[:1]

    poison_path = os.path.join(os.path.dirname(OUT), "caption_poison.json")
    breadcrumb = os.path.join(os.path.dirname(OUT), "caption_attempting.txt")
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

        import numpy as np
        import librosa
        import torch
        from transformers import (AutoProcessor, BitsAndBytesConfig,
                                  MusicFlamingoForConditionalGeneration)

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
            if YIELD_BELOW_S:
                sys.path.insert(0, f"{BASE}/radio")
                import stations
                level = stations.tank_level(stations.active())
                if level < YIELD_BELOW_S:
                    print(f"YIELD tank at {level:.0f}s < {YIELD_BELOW_S:.0f}s "
                          "— handing the GPU back to generation", flush=True)
                    sys.exit(4)
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
    finally:
        try:
            os.unlink(PAUSE_FLAG)
            print("PAUSE lowered", flush=True)
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
