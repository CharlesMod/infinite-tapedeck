#!/usr/bin/env python3
"""Deterministic per-track axes for the corpus: tempo, key, energy shape,
spectral character, onset density. Append-safe: keyed by relative path +
content hash, so resyncs only process new or changed files.

Usage: venv/bin/python analysis/features.py [library_dir] [out_jsonl]
"""
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import librosa
import numpy as np

import os as _os
BASE = _os.environ.get("TAPEDECK_BASE") or _os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))


LIBRARY = sys.argv[1] if len(sys.argv) > 1 else f"{BASE}/library"
OUT = sys.argv[2] if len(sys.argv) > 2 else f"{BASE}/analysis/features.jsonl"
EXTS = (".flac", ".mp3", ".m4a", ".ogg", ".opus", ".wav", ".wma", ".aac", ".aiff")
SR = 22050

# Krumhansl-Schmuckler key profiles
MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def content_hash(path):
    h = hashlib.blake2b(digest_size=16)
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def estimate_key(chroma_mean):
    best = (-2.0, None)
    for shift in range(12):
        prof = np.roll(chroma_mean, -shift)
        for name, ref in (("major", MAJOR), ("minor", MINOR)):
            r = np.corrcoef(prof, ref)[0, 1]
            if r > best[0]:
                best = (r, f"{NOTES[shift]} {name}")
    return {"key": best[1], "key_corr": round(float(best[0]), 3)}


def analyze(path):
    y, sr = librosa.load(path, sr=SR, mono=True)
    dur = len(y) / sr
    if dur < 10:
        raise ValueError(f"too short: {dur:.1f}s")

    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)

    rms = librosa.feature.rms(y=y)[0]
    thirds = np.array_split(rms, 3)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    flatness = librosa.feature.spectral_flatness(y=y)[0]
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)

    f = {
        "duration_s": round(dur, 1),
        "tempo_bpm": round(float(np.atleast_1d(tempo)[0]), 1),
        "beat_count": int(len(beats)),
        "onset_per_s": round(len(onsets) / dur, 2),
        "rms_mean": round(float(rms.mean()), 4),
        "rms_std": round(float(rms.std()), 4),
        # start/mid/end energy: does it build, plateau, dissolve
        "energy_thirds": [round(float(t.mean()), 4) for t in thirds],
        "centroid_hz": round(float(centroid.mean()), 0),
        "centroid_std": round(float(centroid.std()), 0),
        "rolloff_hz": round(float(rolloff.mean()), 0),
        "flatness": round(float(flatness.mean()), 4),
    }
    f.update(estimate_key(chroma.mean(axis=1)))
    return f


def _worker_init():
    # one thread per worker: N workers x M BLAS/numba threads would thrash
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMBA_NUM_THREADS"):
        os.environ[v] = "1"
    try:
        import threadpoolctl
        threadpoolctl.threadpool_limits(1)
    except Exception:
        pass


def _one(path, done):
    rel = os.path.relpath(path, LIBRARY)
    try:
        h = content_hash(path)
        if (rel, h) in done:
            return "skip", rel, None
        return "ok", rel, {"path": rel, "hash": h, **analyze(path)}
    except Exception as e:
        return "fail", rel, f"{e!r:.100}"


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    done = set()
    if os.path.exists(OUT):
        with open(OUT) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    done.add((rec["path"], rec["hash"]))
                except Exception:
                    pass

    todo = []
    for root, _, files in os.walk(LIBRARY):
        for name in sorted(files):
            if name.lower().endswith(EXTS):
                todo.append(os.path.join(root, name))
    print(f"{len(todo)} audio files, {len(done)} already analyzed", flush=True)

    # Tracks are independent and librosa is CPU-bound, so analyse them in a
    # process pool. Only this process writes the jsonl, so the file stays
    # append-safe. TAPEDECK_FEATURE_WORKERS overrides the worker count.
    # Leave one physical core (both hyperthreads) to the OS: pin this process,
    # and so every worker it forks, to the other logical CPUs.
    # TAPEDECK_RESERVE_CPU names the core to keep free (default: cpu0's core).
    cpus = set(os.sched_getaffinity(0))
    try:
        keep = os.environ.get("TAPEDECK_RESERVE_CPU", "0")
        with open(f"/sys/devices/system/cpu/cpu{keep}/topology/thread_siblings_list") as f:
            reserved = set()
            for part in f.read().strip().split(","):
                a, _, b = part.partition("-")
                reserved |= set(range(int(a), int(b or a) + 1))
        if cpus - reserved:
            cpus -= reserved
            os.sched_setaffinity(0, cpus)
            print(f"reserved cpus {sorted(reserved)} for the OS", flush=True)
    except (OSError, ValueError):
        pass
    workers = int(os.environ.get("TAPEDECK_FEATURE_WORKERS") or len(cpus))
    print(f"{workers} worker processes on {len(cpus)} cpus", flush=True)
    ok = fail = skip = 0
    with open(OUT, "a") as out, ProcessPoolExecutor(
            max_workers=workers, initializer=_worker_init) as pool:
        futs = [pool.submit(_one, path, done) for path in todo]
        for i, fut in enumerate(as_completed(futs), 1):
            kind, rel, payload = fut.result()
            if kind == "skip":
                skip += 1
            elif kind == "ok":
                out.write(json.dumps(payload) + "\n")
                out.flush()
                ok += 1
            else:
                print(f"FAIL {rel}: {payload}", flush=True)
                fail += 1
            print(f"PROG {i} {len(todo)} {rel}", flush=True)
            if i % 25 == 0:
                print(f"[{i}/{len(todo)}] ok={ok} skip={skip} fail={fail}", flush=True)
    print(f"done: ok={ok} skip={skip} fail={fail}", flush=True)
    print(f"RESULT {ok} analyzed"
          + (f", {skip} already cached" if skip else "")
          + (f", {fail} failed" if fail else ""), flush=True)
    if fail and not ok and not skip:
        # Nothing analyzed and nothing cached: clustering would later fail on
        # an empty features.jsonl, far from the actual cause (see FAIL lines).
        print("FEATURES FAILED: every track failed to analyze. librosa needs "
              "ffmpeg on PATH to decode mp3/m4a/opus; run "
              "analysis/preflight.py to check the environment.", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
