#!/usr/bin/env python3
"""CLAP embeddings per track: 10-s windows embedded individually, mean-pooled
per track (normalized). Windows are kept for later within-track dynamics.
GPU when both ComfyUI queues are idle, else CPU. Resume-safe.

Usage: venv/bin/python analysis/embed.py [library_dir] [out_dir]
"""
import hashlib
import json
import os
import sys
import time
import urllib.request

import librosa
import numpy as np
import torch
from transformers import ClapModel, ClapProcessor

import os as _os
BASE = _os.environ.get("TAPEDECK_BASE") or _os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))


LIBRARY = sys.argv[1] if len(sys.argv) > 1 else f"{BASE}/library"
OUTDIR = sys.argv[2] if len(sys.argv) > 2 else f"{BASE}/analysis"
EXTS = (".flac", ".mp3", ".m4a", ".ogg", ".opus", ".wav", ".wma", ".aac", ".aiff")
SR = 48000
WIN_S = 10
# NOT larger_clap_music: that repo serves collapsed weights (Aug 2026) — every
# embedding lands within cosine ~0.999 of every other. htsat-unfused verified:
# matching caption +0.51, mismatched +0.06, noise 0.04.
MODEL = "laion/clap-htsat-unfused"
SERVERS = ("http://127.0.0.1:8188", "http://127.0.0.1:8189")


def gpus_busy():
    """A generation on either server owns the card; we wait."""
    for host in SERVERS:
        try:
            with urllib.request.urlopen(host + "/queue", timeout=3) as r:
                q = json.load(r)
            if q.get("queue_running") or q.get("queue_pending"):
                return True
        except Exception:
            pass  # server down = not busy
    return False


def pick_device():
    # the PAUSE flag means the captioner (or Dean) owns the card outside
    # the ComfyUI queues — CPU is slower but never collides
    if os.path.exists(f"{BASE}/radio/PAUSE"):
        return "cpu"
    if torch.cuda.is_available() and not gpus_busy():
        return "cuda"
    return "cpu"


def main():
    os.makedirs(os.path.join(OUTDIR, "clap_windows"), exist_ok=True)
    index_path = os.path.join(OUTDIR, "clap_index.json")
    index = {}
    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)

    todo = []
    for root, _, files in os.walk(LIBRARY):
        for name in sorted(files):
            if name.lower().endswith(EXTS):
                rel = os.path.relpath(os.path.join(root, name), LIBRARY)
                if rel not in index:
                    todo.append(rel)
    print(f"{len(todo)} tracks to embed ({len(index)} already done)", flush=True)
    if not todo:
        return

    device = pick_device()
    print(f"device: {device}", flush=True)
    model = ClapModel.from_pretrained(MODEL).to(device).eval()
    processor = ClapProcessor.from_pretrained(MODEL)

    failures = []
    for i, rel in enumerate(todo, 1):
        # politeness: if a generation started, hop off the card
        if device == "cuda" and i % 5 == 0 and gpus_busy():
            print("generation started — moving to CPU", flush=True)
            model.to("cpu")
            device = "cpu"
        try:
            y, _ = librosa.load(os.path.join(LIBRARY, rel), sr=SR, mono=True)
            n = max(1, int(len(y) / (SR * WIN_S)))
            wins = [y[j * SR * WIN_S:(j + 1) * SR * WIN_S] for j in range(n)]
            wins = [w for w in wins if len(w) >= SR * 3] or [y[:SR * WIN_S]]
            vecs = []
            for w in wins:
                inp = processor(audio=w, sampling_rate=SR, return_tensors="pt")
                inp = {k: v.to(device) for k, v in inp.items()}
                with torch.no_grad():
                    out = model.get_audio_features(**inp).pooler_output
                    vecs.append(out.squeeze(0).cpu().numpy())
            arr = np.stack(vecs)
            # hashed name: flattened paths can exceed the 255-byte filename limit
            safe = hashlib.blake2b(rel.encode(), digest_size=12).hexdigest() + ".npy"
            np.save(os.path.join(OUTDIR, "clap_windows", safe), arr)
            mean = arr.mean(axis=0)
            index[rel] = {"windows": len(vecs), "file": safe,
                          "norm": round(float(np.linalg.norm(mean)), 4)}
            if i % 10 == 0 or i == len(todo):
                with open(index_path, "w") as f:
                    json.dump(index, f, indent=1)
                print(f"[{i}/{len(todo)}]", flush=True)
        except Exception as e:
            failures.append((rel, repr(e)[:200]))
            print(f"FAIL {rel}: {e!r:.100}", flush=True)
        print(f"PROG {i} {len(todo)} {rel}", flush=True)

    with open(index_path, "w") as f:
        json.dump(index, f, indent=1)

    # pooled matrix, aligned to sorted index keys
    keys = sorted(index.keys())
    if not keys:
        # Every track failed. Left alone this used to surface three lines
        # later as "ValueError: need at least one array to stack", which
        # tells the user nothing — the real news is the first failure.
        print(f"\nEMBEDDING FAILED: 0 of {len(todo)} tracks embedded.",
              flush=True)
        if failures:
            print(f"first failure: {failures[0][0]}\n  {failures[0][1]}",
                  flush=True)
            print("\nA failure on every track is usually one of:\n"
                  "  - audio decoding: librosa needs ffmpeg on PATH for mp3/"
                  "m4a/opus\n"
                  "  - a partial install: torch, transformers and librosa "
                  "must live in\n"
                  "    the SAME interpreter that runs this script\n"
                  "  - the CLAP weights failed to download (needs network on "
                  "first run)\n"
                  "Run analysis/preflight.py to check all of the above.",
                  flush=True)
        sys.exit(1)
    mat = []
    for k in keys:
        fname = index[k].get("file", k.replace(os.sep, "__") + ".npy")
        arr = np.load(os.path.join(OUTDIR, "clap_windows", fname))
        m = arr.mean(axis=0)
        mat.append(m / (np.linalg.norm(m) + 1e-9))
    np.save(os.path.join(OUTDIR, "embeddings.npy"), np.stack(mat))
    with open(os.path.join(OUTDIR, "embeddings_keys.json"), "w") as f:
        json.dump(keys, f, indent=1)
    print(f"pooled matrix: {len(keys)} x {mat[0].shape[0]}", flush=True)


if __name__ == "__main__":
    main()
