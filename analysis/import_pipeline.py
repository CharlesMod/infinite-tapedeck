#!/usr/bin/env python3
"""Library capture, end to end: point at a music directory and run the whole
listener pipeline — inventory → features → embeddings → (captions) → veins →
starter essence cards. What we did by hand for the first corpus, as one
resumable command.

Progress is written to analysis/import_progress.json after every parsed
event, so a UI can draw a per-stage track bar and an overall capture bar.
Stage scripts are resume-safe, so re-running an import only processes new or
changed files.

Usage: import_pipeline.py --source DIR [--with-captions]
"""
import json
import os
import signal
import subprocess
import sys
import time

BASE = os.environ.get("TAPEDECK_BASE") or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = f"{BASE}/analysis"
PY = f"{BASE}/venv/bin/python"
PROGRESS = f"{SCRIPTS}/import_progress.json"

sys.path.insert(0, f"{BASE}/radio")
import stations  # noqa: E402

WITH_CAPTIONS = "--with-captions" in sys.argv
STATION = None
if "--station" in sys.argv:
    STATION = sys.argv[sys.argv.index("--station") + 1]
elif "--source" in sys.argv:
    STATION = stations.LEGACY_SLUG  # bare --source keeps the old behavior

P = stations.paths(STATION) if STATION else None
if not P:
    print("usage: import_pipeline.py --station SLUG [--with-captions]")
    sys.exit(2)
SRC = os.path.realpath(os.path.expanduser(
    sys.argv[sys.argv.index("--source") + 1] if "--source" in sys.argv
    else P["source"]))
A = P["analysis"]
if not os.path.isdir(SRC):
    print(f"source is not a directory: {SRC}")
    sys.exit(2)
os.makedirs(A, exist_ok=True)

_child = None
_state = {
    "state": "running", "source": SRC, "with_captions": WITH_CAPTIONS,
    "station": STATION,
    "pid": os.getpid(), "started": int(time.time()),
    "stages": [], "stage_idx": -1, "overall_pct": 0.0,
    "track_i": 0, "track_n": 0, "current_track": "", "log": [],
}


def flush():
    tmp = PROGRESS + ".tmp"
    with open(tmp, "w") as f:
        json.dump(_state, f)
    os.replace(tmp, PROGRESS)


def note(line):
    _state["log"] = (_state["log"] + [line])[-25:]
    print(line, flush=True)
    flush()


def _sigterm(*_):
    _state["state"] = "cancelled"
    if _child and _child.poll() is None:
        _child.terminate()
    flush()
    sys.exit(143)


signal.signal(signal.SIGTERM, _sigterm)


def other_captioner_running():
    """Someone else holds the PAUSE flag. The captioner writes its pid into
    the flag; a live pid = a running captioner. An unparseable flag is a
    manual hold (Dean's `touch PAUSE`) — treat as busy, never load 11 GB
    over it. A dead pid is a stale flag from a crash — not busy."""
    flag = f"{BASE}/radio/PAUSE"
    if not os.path.exists(flag):
        return False
    try:
        pid = int(open(flag).read().strip())
    except (ValueError, OSError):
        return True
    try:
        os.kill(pid, 0)
        return pid != os.getpid() and not _is_our_child(pid)
    except OSError:
        return False


def _is_our_child(pid):
    try:
        with open(f"/proc/{pid}/stat") as f:
            return int(f.read().split()[3]) == os.getpid()
    except Exception:
        return False


def auto_cards():
    """Starter essence cards from vein statistics — enough to run the radio.
    Never overwrites a hand-tuned essence_cards.json; writes the _auto file
    always, and bootstraps the real one only when absent."""
    with open(f"{A}/veins.json") as f:
        veins = json.load(f)["veins"]
    captions = {}
    cap_path = f"{A}/captions.jsonl"
    if os.path.exists(cap_path):
        with open(cap_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    captions[r["path"]] = r["caption"]
                except Exception:
                    continue

    total = sum(v["size"] for v in veins.values()) or 1
    cards = {}
    for label, v in veins.items():
        vocal = v["vocal_share"]
        vocals = ("lead vocals with lyrics are central to this vein"
                  if vocal > 0.5 else
                  "occasional vocals; default instrumental" if vocal > 0.15
                  else "none — fully instrumental")
        arc = v["energy_arc"]
        arc_desc = ("builds to a mid peak and lands soft"
                    if arc[1] >= arc[0] and arc[1] >= arc[2]
                    else "builds steadily to the end"
                    if arc[2] >= arc[1] >= arc[0]
                    else "front-loaded, easing off")
        artists = ", ".join(a for a, _ in v["top_artists"][:4])
        excerpt = ""
        for t in v["central_tracks"]:
            if t in captions:
                excerpt = captions[t][:400]
                break
        essence = (f"{v['size']} tracks. Median {v['bpm']['median']:.0f} BPM "
                   f"(IQR {v['bpm']['iqr'][0]:.0f}–{v['bpm']['iqr'][1]:.0f}), "
                   f"spectral centroid ~{v['centroid_hz_median']:.0f} Hz, "
                   f"{v['onset_per_s_median']:.1f} onsets/s; energy {arc_desc}. "
                   f"Signature artists: {artists}.")
        if excerpt:
            essence += f" A central track, described by ear: {excerpt}"
        cards[label] = {
            "name": f"Vein {label}",
            "weight": round(v["size"] / total, 3),
            "essence": essence,
            "envelope": {
                "bpm": [v["bpm"]["iqr"][0], v["bpm"]["iqr"][1]],
                "bpm_median": v["bpm"]["median"],
                "spectral": f"~{v['centroid_hz_median']:.0f} Hz centroid",
                "onset_per_s": v["onset_per_s_median"],
                "energy_arc": arc_desc,
                "keys": ", ".join(k for k, _ in v["top_keys"]),
                "length_s": [max(60, v["duration_median_s"] * 0.7),
                             min(300, v["duration_median_s"] * 1.3)],
            },
            "vocals": vocals,
            "mutation_axes": ["lead instrument or texture", "tempo within envelope",
                              "arrangement density", "percussion character"],
            "fixed_core": ["the vein's overall mood and energy shape"],
            "caption_seed": (
                f"Global Metadata: {arc_desc} instrumental piece, "
                f"{v['bpm']['median']:.0f} BPM, {v['top_keys'][0][0]}, "
                f"true to the vein's character.\n\nVocal Details: "
                f"{'Lead vocal carries the song.' if vocal > 0.5 else 'No vocals, fully instrumental.'}"
                f"\n\nArrangement: Opens sparse, {arc_desc}, closes gently."),
            "auto": True,
        }
    with open(f"{A}/essence_cards_auto.json", "w") as f:
        json.dump({"_meta": {"generated": "import_pipeline auto_cards",
                             "note": "starter cards from stats — refine by hand or LLM"},
                   "veins": cards}, f, indent=1)
    if not os.path.exists(f"{A}/essence_cards.json"):
        with open(f"{A}/essence_cards.json", "w") as f:
            json.dump({"_meta": {"generated": "auto (bootstrap)"},
                       "veins": cards}, f, indent=1)
        note("cards: bootstrapped essence_cards.json from stats")
    else:
        note("cards: essence_cards.json exists — wrote essence_cards_auto.json only")


STAGES = [
    ("inventory", [PY, f"{SCRIPTS}/inventory.py", SRC, f"{A}/inventory.json"], 3, False),
    ("features", [PY, f"{SCRIPTS}/features.py", SRC, f"{A}/features.jsonl"], 40, True),
    ("embeddings", [PY, f"{SCRIPTS}/embed.py", SRC, A], 30, True),
    ("captions", [PY, f"{SCRIPTS}/caption_pass.py", "--source", SRC,
                  "--out", f"{A}/captions.jsonl",
                  "--yield-below", "900"], 120, True),
    ("veins", [PY, f"{SCRIPTS}/cluster.py", SRC, A], 5, False),
    ("cards", None, 2, False),  # in-process
]


def main():
    global _child
    active = [(n, c, w, t) for n, c, w, t in STAGES
              if n != "captions" or WITH_CAPTIONS]
    _state["stages"] = [{"name": n, "state": "pending"} for n, *_ in active]
    total_w = sum(w for _, _, w, _ in active)
    done_w = 0.0
    flush()

    for idx, (name, cmd, weight, tracked) in enumerate(active):
        _state["stage_idx"] = idx
        _state["stages"][idx]["state"] = "running"
        _state["track_i"] = _state["track_n"] = 0
        _state["current_track"] = ""
        note(f"=== stage: {name} ===")

        if name == "captions" and other_captioner_running():
            _state["stages"][idx]["state"] = "skipped"
            note("captions: another captioner is running — skipped "
                 "(re-run import later to fill in)")
            done_w += weight
            continue

        if cmd is None:  # cards
            try:
                auto_cards()
                _state["stages"][idx]["state"] = "done"
            except Exception as e:
                _state["stages"][idx]["state"] = "failed"
                note(f"cards failed: {e!r:.120}")
            done_w += weight
            _state["overall_pct"] = round(done_w / total_w * 100, 1)
            flush()
            continue

        # generation-vs-captions duty cycle: the captioner exits 4 when the
        # radio's tank drops below its yield threshold; we wait for the tank
        # daemon to refill past RESUME_ABOVE_S, then hand the card back.
        RESUME_ABOVE_S = 1800
        attempts = 3 if name == "captions" else 1
        crash_count = 0
        rc = 1
        while True:
            _child = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, text=True)
            for line in _child.stdout:
                line = line.rstrip()
                if line.startswith("PROG "):
                    try:
                        _, i, n, rel = line.split(" ", 3)
                        _state["track_i"], _state["track_n"] = int(i), int(n)
                        _state["current_track"] = rel
                        frac = int(i) / max(1, int(n))
                        _state["overall_pct"] = round(
                            (done_w + weight * frac) / total_w * 100, 1)
                        flush()
                    except ValueError:
                        pass
                elif line:
                    note(f"[{name}] {line[:160]}")
            rc = _child.wait()
            _child = None
            if rc == 4:
                # yielded to the radio; wait for the tank to refill, then
                # resume. Yields are normal operation, not failures.
                note(f"[{name}] yielded to radio — resuming when tank "
                     f"≥ {RESUME_ABOVE_S}s")
                # SIGTERM exits the process outright, so a plain loop is
                # correct here (a `_stop` flag was the daemon's idiom, and
                # borrowing it uninitialized killed every capture at first yield)
                while stations.tank_level(stations.active()) < RESUME_ABOVE_S:
                    time.sleep(60)
                note(f"[{name}] tank refilled — taking the GPU back")
                continue
            # exit 3 = poison recorded by the script; negative = native crash
            # (SIGABRT et al) whose victim the breadcrumb identifies on the
            # next start. Either way a fresh process resumes past it.
            if rc != 3 and rc >= 0:
                break
            crash_count += 1
            if crash_count >= attempts:
                note(f"[{name}] crashed {crash_count}x — giving up this stage")
                break
            note(f"[{name}] crashed (rc={rc}) — relaunching "
                 f"({crash_count + 1}/{attempts})")
        _state["stages"][idx]["state"] = "done" if rc == 0 else "failed"
        if rc != 0:
            _state["state"] = "failed"
            note(f"{name} exited {rc} — import stopped")
            flush()
            sys.exit(1)
        done_w += weight
        _state["overall_pct"] = round(done_w / total_w * 100, 1)
        flush()

    _state["state"] = "done"
    _state["overall_pct"] = 100.0
    note("import complete")
    flush()


if __name__ == "__main__":
    main()
