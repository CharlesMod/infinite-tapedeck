#!/usr/bin/env python3
"""Library capture, end to end: point at a music directory and run the whole
listener pipeline — inventory → features → embeddings → (captions) → veins →
starter essence cards. What we did by hand for the first corpus, as one
resumable command.

Progress is written to analysis/import_progress.json after every parsed
event, so a UI can draw a per-stage track bar and an overall capture bar.
Stage scripts are resume-safe, so re-running an import only processes new or
changed files.

Usage: import_pipeline.py --station SLUG [--no-captions] [--describe TEXT]
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

USAGE = ("usage: import_pipeline.py --station SLUG [--no-captions] "
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


# Captions are the default: a station described only by tempo and brightness
# numbers gives the generator nothing to aim at, and everything downstream is
# written from that description. --with-captions is still accepted so older
# instructions keep working.
WITH_CAPTIONS = "--no-captions" not in sys.argv
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


# ------------------------------------------------------------ presentation
#
# Two sinks with different jobs. The console (and so the deck's capture log,
# which is this process's stdout) gets one legible line per stage. The full
# child output goes to analysis/capture.log for bug reports. Per-call library
# warnings are counted rather than repeated: 2.9M lines and 240 MB of log
# were measured on one real corpus, which also pushed every useful line out
# of the deck's 25-line readout.

TTY = sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def _sgr(code):
    return (lambda s: f"\033[{code}m{s}\033[0m") if TTY else (lambda s: str(s))


BOLD, DIM, GREEN, YELLOW, RED = (_sgr("1"), _sgr("2"), _sgr("32"),
                                 _sgr("33"), _sgr("31"))

CAPTURE_LOG = f"{A}/capture.log"
_logf = None
_painted = False
_seen_noise = {}

NOISE_KEYS = ("MatMul8bitLt:", "Loading weights:", "Loading checkpoint",
              "You are sending unauthenticated requests", "it/s]", "s/it]")
PROBLEM_KEYS = ("FAIL", "UNDECODABLE", "Traceback", "NOT ENOUGH", "POISON",
                "YIELD", "not enough", "Error", "error:")


def raw(line):
    """Full detail, for bug reports. Never pretty, never truncated.

    Flushed per line on purpose: the moment you most want this file is while
    a capture is still running — a stage that looks hung, a pass quietly
    failing every track — and a buffered log is empty exactly then."""
    if _logf:
        _logf.write(line + "\n")
        _logf.flush()


def is_repeat_noise(line):
    """First of each kind is kept; the rest are counted and dropped."""
    for key in NOISE_KEYS:
        if key in line:
            _seen_noise[key] = _seen_noise.get(key, 0) + 1
            return _seen_noise[key] > 1
    return False


def is_problem(line):
    return any(p in line for p in PROBLEM_KEYS)


def paint(text):
    """Transient progress: terminal only, never reaches any log."""
    global _painted
    if TTY:
        sys.stdout.write("\r\033[2K" + text[:118])
        sys.stdout.flush()
        _painted = True


def say(line=""):
    """A permanent console line."""
    global _painted
    if _painted:
        sys.stdout.write("\r\033[2K")
        _painted = False
    print(line, flush=True)


def bar(frac, width=16):
    filled = max(0, min(width, int(round(frac * width))))
    return "█" * filled + "·" * (width - filled)


def human(sec):
    if sec < 60:
        return f"{sec:.1f}s"
    m, s = divmod(int(sec), 60)
    return f"{m}m {s:02d}s" if m < 60 else f"{m // 60}h {m % 60:02d}m"


def note(line):
    """A real event: the deck's log ring, the console, and the full log."""
    _state["log"] = (_state["log"] + [line])[-25:]
    raw(line)
    say(line)
    flush()


def done(name, summary, secs, mark=None):
    """One line per stage — the whole point of the console output."""
    mark = mark or GREEN("✓")
    line = f"  {mark} {BOLD(f'{name:<11}')} {summary or 'done'}"
    pad = max(1, 74 - len(f"  x {name:<11} {summary or 'done'}"))
    say(line + " " * pad + DIM(human(secs)))
    _state["log"] = (_state["log"] + [f"{name}: {summary}"])[-25:]
    raw(f"--- {name}: {summary} ({human(secs)})")
    flush()


def _sigterm(*_):
    _state["state"] = "cancelled"
    if _child and _child.poll() is None:
        _child.terminate()
    flush()
    sys.exit(143)


signal.signal(signal.SIGTERM, _sigterm)


def pid_alive(pid):
    """POSIX probes liveness with signal 0; Windows os.kill() would call
    TerminateProcess() instead and kill what it is asking about."""
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if os.name == "nt":
        import ctypes
        handle = ctypes.windll.kernel32.OpenProcess(0x00100000, False, pid)
        if not handle:
            return False
        ctypes.windll.kernel32.CloseHandle(handle)
        return True
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


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
    if not pid_alive(pid):
        return False
    return pid != os.getpid() and not _is_our_child(pid)


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
    n = len(cards)
    noun = f"{n} card" + ("" if n == 1 else "s")
    if not os.path.exists(f"{A}/essence_cards.json"):
        with open(f"{A}/essence_cards.json", "w") as f:
            json.dump({"_meta": {"generated": "auto (bootstrap)"},
                       "veins": cards}, f, indent=1)
        return (f"{noun} written "
                + ("from listening notes" if captions else "from statistics"))
    return f"{noun} refreshed as _auto (yours kept)"


AUDIO_EXTS = (".flac", ".mp3", ".m4a", ".ogg", ".opus", ".wav", ".wma",
              ".aac", ".aiff")
CAPTION_S_PER_TRACK = 15  # measured on an RTX 5080, 8-bit Flamingo
FLAMINGO_GB = 17


YIELD_WAIT_MAX_S = 3600  # backstop: never park a capture for longer than this


def radio_is_live():
    """Is a tank daemon actually running? The daemon rewrites
    daemon_state.json on every log line, so a fresh timestamp is a real
    heartbeat. Without this, a first capture yields the GPU to a radio that
    does not exist and waits forever for a tank nothing will fill."""
    try:
        with open(f"{BASE}/radio/daemon_state.json") as f:
            return time.time() - int(json.load(f).get("ts", 0)) < 300
    except (OSError, ValueError, TypeError, AttributeError):
        return False


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


def caption_blockers():
    """Why the listening pass cannot run, or [] if it can.

    Captions are the default, but a missing dependency must never cost the
    user their whole capture — the radio still runs without them. Blockers
    downgrade to a warning, they do not stop the pipeline."""
    out = []
    missing = missing_caption_deps()
    if missing:
        out.append(f"missing {', '.join(missing)} "
                   f"(pip install {' '.join(missing)})")
    if not flamingo_cached():
        free_gb = shutil.disk_usage(BASE).free / 2 ** 30
        if free_gb < FLAMINGO_GB + 3:
            out.append(f"only {free_gb:.0f} GB disk free, the model needs "
                       f"~{FLAMINGO_GB} GB")
    return out


def announce_captions(n):
    """Captions are on by default — say what that will cost before it runs,
    so a long pass is never a surprise."""
    mins = max(1, round(n * CAPTION_S_PER_TRACK / 60))
    cost = f"~{mins} min of GPU time"
    if not flamingo_cached():
        cost += f", plus a one-time {FLAMINGO_GB} GB model download"
    say(f"  {DIM('AI listening pass on — ' + cost + '.')}")
    say(f"  {DIM('It is what makes generated songs match your library; '
                 'skip with --no-captions.')}")
    say()
    _state["log"] = (_state["log"] + [f"AI listening pass on ({cost})"])[-25:]


def explain_no_captions(n):
    """--no-captions was passed: be precise about what is being given up,
    and offer the cheap alternative."""
    mins = max(1, round(n * CAPTION_S_PER_TRACK / 60))
    say(f"  {YELLOW('!')} {BOLD('No listening pass')} — this station will be "
        "described to the")
    say(f"    generator only by numbers: tempo, brightness, energy shape.")
    say(f"    Nothing says which genre or instruments, so takes drift "
        "off-style.")
    say()
    if description():
        say(f"    {DIM('Using your description: ' + description()[:60])}")
    else:
        say(f"    {DIM('Say it in your own words instead:')}")
        say(f"    {DIM('--describe \"dark melodic techno, analog hardware, '
                       'no vocals\"')}")
    say(f"    {DIM(f'Or drop --no-captions to run the pass (~{mins} min).')}")
    say()


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


def header(n_audio):
    say()
    say(f"  {BOLD('∞ TAPEDECK')} {DIM('·')} capturing {BOLD(STATION)}")
    say(f"  {DIM(f'{n_audio} tracks · {SRC}')}")
    say()


def footer(secs):
    suppressed = sum(n - 1 for n in _seen_noise.values() if n > 1)
    folded = (f"  ({suppressed} repeated library message"
              f"{'' if suppressed == 1 else 's'} folded away)")
    say()
    say(f"  {GREEN('Captured')} in {human(secs)}."
        + (DIM(folded) if suppressed else ""))
    if not stations_ready():
        say(f"  {DIM('Full detail: ' + CAPTURE_LOG)}")
        return
    say()
    say(f"  {BOLD('Next:')} start the radio, then open the deck")
    say(f"    {sys.executable} radio/tank_daemon.py")
    say(f"    {DIM('http://<host>:8188/extensions/music_studio/index.html')}")
    say()


def stations_ready():
    return os.path.exists(os.path.join(A, "essence_cards.json"))


def main():
    global _child, WITH_CAPTIONS, _logf
    t_run = time.time()
    n_audio = count_audio(SRC)
    try:
        _logf = open(CAPTURE_LOG, "w")
    except OSError:
        _logf = None
    header(n_audio)
    if WITH_CAPTIONS:
        blockers = caption_blockers()
        if blockers:
            WITH_CAPTIONS = False
            _state["with_captions"] = False
            note("AI listening pass unavailable: " + "; ".join(blockers))
            note("continuing without it — the radio will still run, but "
                 "takes will drift off-style. See docs/INSTALL.md step 6.")
        elif n_audio:
            announce_captions(n_audio)
    elif n_audio:
        explain_no_captions(n_audio)
        _state["log"] = (_state["log"] + ["captions off by request — "
                                          "statistics only"])[-25:]

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
        raw(f"\n=== stage: {name} ===")
        t_stage = time.time()
        summary = ""
        paint(f"  {DIM('▸')} {name:<11} {DIM('starting…')}")

        if name == "captions" and other_captioner_running():
            _state["stages"][idx]["state"] = "skipped"
            done(name, "skipped — another captioner holds the card",
                 time.time() - t_stage, mark=YELLOW("–"))
            done_w += weight
            continue

        if cmd is None:  # cards
            try:
                summary = auto_cards()
                _state["stages"][idx]["state"] = "done"
                done(name, summary, time.time() - t_stage)
            except Exception as e:
                _state["stages"][idx]["state"] = "failed"
                done(name, f"failed: {e!r:.100}", time.time() - t_stage,
                     mark=RED("✗"))
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
                        paint(f"  {DIM('▸')} {name:<11} {bar(frac)} "
                              f"{i}/{n}  {DIM(os.path.basename(rel)[:44])}")
                    except ValueError:
                        pass
                elif line.startswith("RESULT "):
                    summary = line[7:].strip()
                    raw(line)
                elif not line:
                    continue
                elif is_repeat_noise(line):
                    raw(line)  # first of its kind only
                elif is_problem(line):
                    note(f"  {YELLOW('!')} {name}: {line[:150]}")
                else:
                    raw(line)  # detail lives in capture.log, not on screen
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
                waited = 0
                while stations.tank_level(stations.active()) < RESUME_ABOVE_S:
                    # Never wait on a tank nothing is filling. A daemon that
                    # died, or was never started, would otherwise park the
                    # capture here forever.
                    if not radio_is_live():
                        note(f"[{name}] no radio running — taking the card "
                             "back instead of waiting")
                        break
                    if waited >= YIELD_WAIT_MAX_S:
                        note(f"[{name}] tank still low after "
                             f"{waited // 60} min — resuming anyway")
                        break
                    time.sleep(60)
                    waited += 60
                else:
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
            done(name, f"failed (exit {rc}) — capture stopped",
                 time.time() - t_stage, mark=RED("✗"))
            say()
            say(f"  {RED('Capture stopped.')} Full detail: {CAPTURE_LOG}")
            flush()
            sys.exit(1)
        done(name, summary, time.time() - t_stage)
        done_w += weight
        _state["overall_pct"] = round(done_w / total_w * 100, 1)
        flush()

    _state["state"] = "done"
    _state["overall_pct"] = 100.0
    flush()
    footer(time.time() - t_run)

    if not WITH_CAPTIONS and not description():
        # The two ways to give this station words instead of only numbers.
        # Worth repeating at the end: this is the moment before the user
        # starts the radio and forms an opinion of it.
        print("\nThis station has no written description of its sound — only "
              "\nmeasured statistics. Generation will drift off-style. Fix "
              "with either:"
              f"\n  {sys.executable} analysis/import_pipeline.py --station "
              f"{STATION}"
              "\n      (re-run without --no-captions: the AI listening pass, "
              "best quality)"
              f"\n  {sys.executable} analysis/import_pipeline.py --station "
              f"{STATION} --describe \"dark melodic techno, analog hardware, "
              "no vocals\""
              "\n      (one sentence in your own words — instant)\n",
              flush=True)


if __name__ == "__main__":
    main()
