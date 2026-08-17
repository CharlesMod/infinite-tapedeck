#!/usr/bin/env python3
"""Library capture, end to end: point at a music directory and run the whole
listener pipeline — inventory → features → embeddings → (captions) → veins →
starter essence cards. What we did by hand for the first corpus, as one
resumable command.

Progress is written to analysis/import_progress.json after every parsed
event, so a UI can draw a per-stage track bar and an overall capture bar.
Stage scripts are resume-safe, so re-running an import only processes new or
changed files.

Usage: import_pipeline.py --station SLUG [--with-captions] [--describe TEXT]
"""
import json
import os
import shutil
import signal
import subprocess
import sys
import time

BASE = os.environ.get("TAPEDECK_BASE") or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = f"{BASE}/analysis"
# Stages run as subprocesses and must use an interpreter that has the
# analysis deps. A venv inside the repo is the documented layout, but the
# supported install shares ComfyUI's venv — in which case the interpreter
# already running this script is the right one.
PY = (f"{BASE}/venv/bin/python" if os.path.exists(f"{BASE}/venv/bin/python")
      else sys.executable)
PROGRESS = f"{SCRIPTS}/import_progress.json"

sys.path.insert(0, f"{BASE}/radio")
import stations  # noqa: E402

USAGE = ("usage: import_pipeline.py --station SLUG [--with-captions] "
         "[--describe TEXT] [--source DIR]")


def opt(name):
    """Value of --name, or None. A flag given without its value is a typo,
    not a traceback."""
    if name not in sys.argv:
        return None
    i = sys.argv.index(name) + 1
    if i >= len(sys.argv) or sys.argv[i].startswith("--"):
        print(f"{name} needs a value\n{USAGE}")
        sys.exit(2)
    return sys.argv[i]


WITH_CAPTIONS = "--with-captions" in sys.argv
DESCRIBE = opt("--describe")
STATION = opt("--station")
if STATION is None and "--source" in sys.argv:
    STATION = stations.LEGACY_SLUG  # bare --source keeps the old behavior

P = stations.paths(STATION) if STATION else None
if not P:
    print(USAGE)
    sys.exit(2)
if not P["source"] and not opt("--source"):
    # An unregistered slug resolves to an empty source, which realpath()
    # helpfully turns into the current directory — a typo would capture
    # whatever folder you happened to be standing in.
    known = ", ".join(s["slug"] for s in stations.listing()) or "(none)"
    print(f"unknown station: {STATION}\nknown stations: {known}\n"
          "Create one from the deck UI, or name a folder with --source DIR.")
    sys.exit(2)
SRC = os.path.realpath(os.path.expanduser(opt("--source") or P["source"]))
A = P["analysis"]
if not os.path.isdir(SRC):
    print(f"source is not a directory: {SRC}")
    sys.exit(2)
os.makedirs(A, exist_ok=True)
if DESCRIBE:
    with open(f"{A}/description.txt", "w") as f:
        f.write(DESCRIBE.strip() + "\n")

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


def description():
    """The listener's own words for this music, if they gave any.

    Statistics cannot name a genre: a card built from BPM and spectral
    centroid alone describes "112 BPM, 2400 Hz" and the generator is free to
    read that as anything. One human sentence ("dark melodic techno, analog
    hardware, no vocals") anchors every caption written from this card.
    Captions from the AI listening pass do the same job, better and per
    track — this is the floor when they are switched off."""
    for path in (f"{A}/description.txt", f"{SRC}/DESCRIPTION.txt"):
        try:
            with open(path) as f:
                text = f.read().strip()
            if text:
                return text
        except OSError:
            continue
    return ""


def sentence(text):
    """Free-text the user typed, safe to concatenate in front of prose."""
    text = text.strip()
    return text if text.endswith((".", "!", "?", ";")) else text + "."


def auto_cards():
    """Starter essence cards from vein statistics — enough to run the radio.
    Never overwrites a hand-tuned essence_cards.json; writes the _auto file
    always, and bootstraps the real one only when absent."""
    with open(f"{A}/veins.json") as f:
        veins = json.load(f)["veins"]
    desc = description()
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
        # phrased as verb clauses so they read correctly after "a piece that"
        arc_desc = ("builds to a mid peak and lands soft"
                    if arc[1] >= arc[0] and arc[1] >= arc[2]
                    else "builds steadily to the end"
                    if arc[2] >= arc[1] >= arc[0]
                    else "starts strong and eases off")
        artists = ", ".join(a for a, _ in v["top_artists"][:4])
        excerpt = ""
        for t in v["central_tracks"]:
            if t in captions:
                excerpt = captions[t][:400]
                break
        essence = (f"{v['size']} tracks. Median {v['bpm']['median']:.0f} BPM "
                   f"(IQR {v['bpm']['iqr'][0]:.0f}–{v['bpm']['iqr'][1]:.0f}), "
                   f"spectral centroid ~{v['centroid_hz_median']:.0f} Hz, "
                   f"{v['onset_per_s_median']:.1f} onsets/s; energy {arc_desc}.")
        if artists:
            essence += f" Signature artists: {artists}."
        if desc:
            essence = f"{sentence(desc)} {essence}"
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
                f"Global Metadata: {sentence(desc) + ' ' if desc else ''}"
                f"An instrumental piece that {arc_desc}, "
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


AUDIO_EXTS = (".flac", ".mp3", ".m4a", ".ogg", ".opus", ".wav", ".wma",
              ".aac", ".aiff")
# Above this the listening pass stops being a coffee break and becomes an
# overnight job, so the recommendation only fires for small libraries.
CAPTION_RECOMMEND_MAX = 150
CAPTION_S_PER_TRACK = 15  # measured on an RTX 5080, 8-bit Flamingo
FLAMINGO_GB = 17


def count_audio(src):
    n = 0
    for _, _, files in os.walk(src):
        n += sum(1 for f in files if f.lower().endswith(AUDIO_EXTS))
    return n


def flamingo_cached():
    cache = (os.environ.get("HF_HOME")
             or os.path.expanduser("~/.cache/huggingface"))
    return os.path.isdir(os.path.join(
        cache, "hub", "models--nvidia--music-flamingo-2601-hf"))


def missing_caption_deps():
    import importlib.util
    return [m for m in ("bitsandbytes", "accelerate")
            if importlib.util.find_spec(m) is None]


def offer_captions(n):
    """Small library, captions off — the single highest-value switch in the
    whole install, so say so, and at a terminal offer to flip it.

    CLAP embeddings give the critic ears but give the generator no words: a
    stats-only essence card knows your tempo and brightness, not that you
    handed it dub techno. Everything downstream is written from that card,
    which is why a caption-less station drifts off-genre and then fails its
    own critic. The listening pass is what closes the gap."""
    cached = flamingo_cached()
    missing = missing_caption_deps()
    mins = max(1, round(n * CAPTION_S_PER_TRACK / 60))
    free_gb = shutil.disk_usage(BASE).free / 2 ** 30
    print("\n" + "=" * 70)
    print(f"  RECOMMENDED: run the AI listening pass on these {n} tracks")
    print("=" * 70)
    print("  Without it, this station's taste is described only by numbers")
    print("  (tempo, brightness, energy shape) — no genre, no instruments, no")
    print("  production character. Generated takes drift off-style and the")
    print("  critic rejects most of them. With it, every track gets a written")
    print("  description that steers generation. It is the difference between")
    print("  a radio that sounds like your music and one that does not.")
    print(f"\n  Cost: about {mins} min of GPU time"
          + ("" if cached else f", plus a one-time {FLAMINGO_GB} GB download")
          + ".")
    print("  Resumable and interruptible; it yields the GPU to the radio.")
    if missing:
        print(f"\n  Needs: pip install {' '.join(missing)}")
    if not cached and free_gb < FLAMINGO_GB + 3:
        print(f"\n  NOT ENOUGH DISK: {free_gb:.0f} GB free, needs "
              f"~{FLAMINGO_GB + 3} GB.")
        return False
    if missing or not sys.stdin.isatty():
        print("\n  Enable it with:  --with-captions")
        print("=" * 70 + "\n")
        return False
    print("=" * 70)
    try:
        ans = input("  Run the listening pass now? [Y/n] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return ans in ("", "y", "yes")


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
    global _child, WITH_CAPTIONS
    n_audio = count_audio(SRC)
    if not WITH_CAPTIONS and 0 < n_audio <= CAPTION_RECOMMEND_MAX:
        if offer_captions(n_audio):
            WITH_CAPTIONS = True
            _state["with_captions"] = True
        else:
            # one condensed line for the deck's capture log — the full block
            # above only reaches whoever is watching a terminal
            note(f"TIP: {n_audio} tracks — the listening pass would take "
                 f"~{max(1, round(n_audio * CAPTION_S_PER_TRACK / 60))} min "
                 "and is what makes takes sound like this station")

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

    if not WITH_CAPTIONS and not description():
        # The two ways to give this station words instead of only numbers.
        # Worth repeating at the end: this is the moment before the user
        # starts the radio and forms an opinion of it.
        print("\nThis station has no written description of its sound — only "
              "\nmeasured statistics. Generation will drift off-style. Fix "
              "with either:"
              f"\n  {sys.executable} analysis/import_pipeline.py --station "
              f"{STATION} --with-captions"
              "\n      (AI listening pass — best quality, per-track)"
              f"\n  {sys.executable} analysis/import_pipeline.py --station "
              f"{STATION} --describe \"dark melodic techno, analog hardware, "
              "no vocals\""
              "\n      (one sentence in your own words — instant)\n",
              flush=True)


if __name__ == "__main__":
    main()
