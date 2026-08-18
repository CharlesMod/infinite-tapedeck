#!/usr/bin/env python3
"""The tank: keeps a buffer of generated-and-approved radio tracks topped up.

Generation runs slower than realtime (~0.6x at 30 steps), so the radio's
continuity lives here, not in the player: whenever the GPU is otherwise idle,
sample a vein from the essence cards, have Poe's Gemma write a fresh caption
(and lyrics, for the vocal vein), generate on the Music3 server, score the
result against the vein's corpus centroid with CLAP, and bank what passes.

Politeness, in priority order:
  1. Dean interactive (H3 queue busy, or a game holding VRAM) -> hold.
  2. A PAUSE file in this directory -> hold (manual override).
  3. Otherwise the card is ours.

H3 preemption can kill a run mid-generation (by design); that surfaces as a
non-success history status and costs one backoff, nothing more.
"""
import json
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.request

BASE = os.environ.get("TAPEDECK_BASE") or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{BASE}/radio")
import stations  # noqa: E402

PAUSE_FLAG = f"{BASE}/radio/PAUSE"
COMFY_OUT = f"{BASE}/ComfyUI/output"

_cfg = {"comfy_host": "http://127.0.0.1:8188", "sibling_hosts": [],
        "llm_base": "http://127.0.0.1:8080", "llm_model": None,
        "steps": 30, "tank_target_s": 10800}
try:
    with open(f"{BASE}/radio/config.json") as _f:
        _cfg.update(json.load(_f))
except FileNotFoundError:
    pass
M3 = _cfg["comfy_host"]
SIBLINGS = list(_cfg["sibling_hosts"])
# Single-machine default: the LLM shares the one card with the generator;
# bundles are written in batches per residency and banked to disk so a
# residency almost never blocks production. llm_base=None disables the LLM.
LLM_URL = (_cfg["llm_base"] or "").rstrip("/") or None
LLM_MODEL = _cfg["llm_model"]
CAPTION_BATCH = int(_cfg.get("caption_batch", 10))
BUNDLE_QUEUE_TARGET = int(_cfg.get("bundle_queue_target", 20))  # pre-written bundles to keep banked on disk
BUNDLE_FILE = f"{BASE}/radio/bundles.jsonl"
MIN_TAKE_S = int(_cfg.get("min_take_s", 45))  # cull true stubs; length ambition lives in captions
TANK_TARGET_TRACKS = int(_cfg.get("tank_target_tracks", 20))  # generate whenever fewer than this many are banked
LYRIC_CAP = float(_cfg.get("lyric_cap", 0.25))  # hard cap: sung takes ≤ 25% of the banked spool


def lyric_veins(cards):
    return {v for v, c in cards.items()
            if "lyrics" in c.get("vocals", "").lower()}


def cap_excludes(cards, per_n, count):
    """Veins barred from the next pick by the lyric cap."""
    lv = lyric_veins(cards)
    if not lv or count <= 0:
        return set()
    sung = sum(per_n.get(v, 0) for v in lv)
    return lv if sung / count >= LYRIC_CAP else set()


# Instrumental section forms, as the body between intro and outro.
#
# One fixed skeleton used to serve every take: instrumental, instrumental,
# bridge, then padding — the bridge always fourth, five distinct skeletons in
# total across every possible length. With sung takes capped at 25%, that was
# one section form for three quarters of the radio. These tags are executable
# structure, so that was not cosmetic: it was every song having the same plan.
#
# Kept to tags the generator already handles, and each form is a shape rather
# than a list — repeating a form's own cycle keeps it recognisable when a
# longer target needs more sections.
INSTRUMENTAL_FORMS = {
    "arch": ["instrumental", "bridge", "instrumental"],
    # never returns to the same material; rondo deliberately does
    "through_composed": ["instrumental", "interlude", "break", "solo",
                         "breakdown"],
    "build_drop": ["build", "drop", "breakdown", "build", "drop"],
    "rondo": ["instrumental", "interlude", "instrumental", "solo"],
    "ostinato": ["instrumental", "instrumental", "break", "instrumental"],
    "suite": ["instrumental", "bridge", "interlude", "instrumental"],
}


def structural_tags(target_s, form=None, arc=None):
    """Tag-only lyric sheet for instrumentals. The lyric vein outperforms
    because section tags are executable structure and structure is length —
    so every vein gets the scaffold, without putting words in its mouth."""
    mins = max(1, int(target_s) // 60)
    name = form or random.choice(list(INSTRUMENTAL_FORMS))
    cycle = INSTRUMENTAL_FORMS[name]
    want = max(2, mins + 1)
    body = [cycle[i % len(cycle)] for i in range(want)]
    # No arc special-casing here. Dropping the [outro] for builds-to-end was
    # tried on the theory that the tag was arguing with the caption, and
    # measured: 5 requested, 0 delivered, no better than before. The cause is
    # upstream — takes end at a median 46% of the length their caption plans
    # for, so a piece told to peak at four minutes is cut off at ninety
    # seconds and never reaches the peak at all. Arc cannot be steered until
    # length can; see sample_arc().
    return "\n\n".join(["[intro]"] + [f"[{s}]" for s in body] + ["[outro]"])
# Adaptive bar (Dean's design): every critic reject eases that vein's bar a
# little so dead air self-limits; the next accept snaps it back to the
# calibrated level. Relief is tactical, never a redefinition of taste.
RELIEF_STEP = float(_cfg.get("relief_step", 0.008))
RELIEF_MAX = float(_cfg.get("relief_max", 0.05))
_relief = {}                  # vein -> current threshold relief

STEPS = int(_cfg["steps"])
# The deck's speed/quality slider writes here, and it is read per take so a
# move applies to the next generation without a restart. Steps are the only
# honest speed lever left: measured 0.86x realtime at 30, and generation time
# is very nearly linear in them. KSampler wants an int, so the slider's 2.5
# granularity is rounded at graph time.
STEPS_FILE = f"{BASE}/radio/speed.json"
STEPS_MIN, STEPS_MAX, STEPS_STEP = 20.0, 40.0, 2.5
# Target song length, also from the deck. 0 means "use each vein's own
# measured length envelope", which is the default and usually the right
# answer — the envelope comes from how long the tracks in that vein actually
# are. A value overrides it for every vein.
LEN_MIN, LEN_MAX, LEN_STEP = 60.0, 300.0, 15.0


def _round_half_up(v):
    """int(round()) is banker's rounding in python: 22.5 -> 22 but 37.5 -> 38.
    The deck's slider uses Math.round, which is always half-up, so the two
    disagreed and the UI promised a step count the daemon did not use."""
    import math
    return int(math.floor(float(v) + 0.5))


def load_target_len():
    """Listener's preferred take length in seconds, or None to use the
    vein's own envelope. Read per take, like the steps slider."""
    try:
        with open(STEPS_FILE) as f:
            v = float(json.load(f).get("target_s") or 0)
        return max(LEN_MIN, min(LEN_MAX, v)) if v else None
    except (OSError, ValueError, TypeError):
        return None


def load_steps():
    try:
        with open(STEPS_FILE) as f:
            v = float(json.load(f)["steps"])
        return max(STEPS_MIN, min(STEPS_MAX, v))
    except (OSError, ValueError, TypeError, KeyError):
        return float(STEPS)
DIT_MODEL = _cfg.get("dit_model", "minimax_music3_dit_fp16.safetensors")
# TWO different guidance knobs, which shared one constant purely because this
# file had one named CFG. KSampler's cfg guides the DiT's diffusion; the text
# encoder's cfg_scale guides the AUTOREGRESSIVE acoustic-token sampler in
# comfy/ldm/minimax_music/ar.py, which is the thing actually choosing notes
# one code at a time. They are unrelated stages and need not agree.
CFG = 1.7                     # diffusion guidance (KSampler)
TOP_K = 50                    # kept as the fallback when jitter is disabled

# The AR sampler's own diversity controls, rolled per take. top_k is how many
# candidate codes are live at each step and cfg_scale is how hard the
# conditioned distribution is pushed from the unconditioned one — between them
# they decide how much freedom the model has choosing each next note, which no
# amount of caption wording can reach. Model defaults are 1.5 / 50; this file
# had been sending 1.7, i.e. tighter adherence and less variety than stock,
# for every take ever generated. Set either range to a pair of equal numbers
# to pin it.
AR_CFG_RANGE = (1.4, 1.8)
# Upper bound pulled back from 120 after measuring stubs against it: 0 stubs
# in 19 takes below top_k 80, 4 in 19 at or above it. The stop token is drawn
# from the same distribution top_k widens, so a wider net makes an early stop
# likelier — plausible mechanism, but n is small and the split is only
# suggestive (Fisher ~0.1), so this is a provisional bound, not a finding.
AR_TOP_K_RANGE = (40, 80)
TANK_TARGET_S = int(_cfg["tank_target_s"])
FOREIGN_VRAM_MB = 3000        # non-ComfyUI VRAM above this = game running
LOOP_SLEEP = 60               # between top-up checks when tank is full/held
BACKOFF = 120                 # after a failed/preempted generation

ONESHOT = bool(os.environ.get("TANK_ONESHOT"))
TEST_SHORT = bool(os.environ.get("TANK_TEST_SHORT"))

_stop = False


def _sigterm(*_):
    global _stop
    _stop = True


# ---------------------------------------------------------------- utilities

def http_json(url, payload=None, timeout=15):
    body = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"} if payload is not None else {}
    req = urllib.request.Request(url, data=body, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
        return json.loads(raw) if raw else {}


STATE_FILE = f"{BASE}/radio/daemon_state.json"


def log(msg):
    """Every log line doubles as the deck's live status readout — the GUI
    must always be able to answer 'why am I waiting?'"""
    print(f"[tank] {msg}", flush=True)
    try:
        tmp = STATE_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"msg": msg, "ts": int(time.time())}, f)
        os.replace(tmp, STATE_FILE)
    except Exception:
        pass


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


def queue_busy(host):
    try:
        q = http_json(host + "/queue", timeout=5)
        return bool(q.get("queue_running")) or bool(q.get("queue_pending"))
    except Exception:
        return None  # server down


def unit_pid(unit):
    try:
        out = subprocess.run(
            ["systemctl", "--user", "show", "-p", "MainPID", "--value", unit],
            capture_output=True, text=True, timeout=5).stdout.strip()
        return int(out) or None
    except Exception:
        return None


def foreign_vram_mb():
    """VRAM held by anything that is not one of the two ComfyUI servers.

    Above threshold with both queues idle means Dean is doing something with
    the card (a game, most likely) — the tank must not shoulder in.
    """
    ours = {unit_pid("comfyui.service"), unit_pid("comfyui-music3.service")}
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return 0
    total = 0
    for line in out.strip().splitlines():
        try:
            pid_s, mem_s = line.split(",")
            pid = int(pid_s)
            if pid in ours:
                continue
            # house tenants coordinate through their own choreography
            # (residencies, PAUSE) — only genuinely foreign load (a game)
            # should hold the radio
            try:
                with open(f"/proc/{pid}/cmdline", "rb") as f:
                    cmd = f.read().decode(errors="replace")
                if "llama-server" in cmd or "caption_pass" in cmd:
                    continue
            except OSError:
                pass
            total += int(mem_s)
        except ValueError:
            continue
    return total


def hold_reason():
    if os.path.exists(PAUSE_FLAG):
        # Failure class: a captioner killed hard (OOM, SIGKILL) leaves its
        # pid-bearing PAUSE behind and the radio would hold forever. A dead
        # pid is debris — clear it and keep playing. Unparseable = Dean's
        # manual touch = sacred.
        try:
            pid = int(open(PAUSE_FLAG).read().strip())
            if pid_alive(pid):
                return "PAUSE held by captioner"
            else:
                os.unlink(PAUSE_FLAG)
                log("stale PAUSE from dead captioner — cleared, resuming")
        except (ValueError, OSError):
            return "PAUSE file present (manual hold)"
    for sib in SIBLINGS:
        if queue_busy(sib):
            return f"sibling busy: {sib}"
    if queue_busy(M3) is None:
        return "music3 server down"
    if foreign_vram_mb() > FOREIGN_VRAM_MB:
        return f"foreign VRAM > {FOREIGN_VRAM_MB} MB (game?)"
    return None


# ------------------------------------------------------------------- lyrics

_llm_down_until = 0.0


_llm_model_cache = None


def _resolve_model():
    global _llm_model_cache
    if LLM_MODEL:
        return LLM_MODEL
    if _llm_model_cache:
        return _llm_model_cache
    try:
        r = http_json(LLM_URL + "/v1/models", timeout=10)
        _llm_model_cache = r["data"][0]["id"]
        return _llm_model_cache
    except Exception:
        return None


def llm_chat(prompt, temperature=0.9, max_tokens=900):
    """First call swaps the model in (fast when page-cached). Thinking is
    disabled where the template honors it, or reasoning models burn the
    whole budget thinking and content comes back empty."""
    if not LLM_URL:
        return None
    model = _resolve_model()
    if not model:
        return None
    for attempt in (1, 2):
        try:
            r = http_json(LLM_URL + "/v1/chat/completions", {
                "model": model, "temperature": temperature,
                "max_tokens": max_tokens,
                "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "user", "content": prompt}],
            }, timeout=240)
            out = r["choices"][0]["message"]["content"].strip()
            if out:
                return out
            raise ValueError("empty content")
        except Exception as e:
            log(f"llm attempt {attempt} failed: {e!r:.80}")
            time.sleep(10)
    _llm_down_until = time.time() + 600
    log("LLM circuit OPEN for 10 min — captions fall back to card seeds")
    return None


def free_music3():
    """Hand the card over: drop the generator's weights and wait for VRAM.
    VERIFIED — loading the LLM onto an unfreed card is how llama-server
    children wedge into 0-tok/s zombies."""
    try:
        http_json(M3 + "/free", {"unload_models": True, "free_memory": True},
                  timeout=15)
    except Exception:
        log("free_music3: /free unreachable — card NOT freed")
        return False
    for _ in range(15):
        try:
            used = int(subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5).stdout.strip())
            if used < 5000:
                return True
        except Exception:
            break
        time.sleep(2)
    log("free_music3: card did NOT free — skipping LLM this residency")
    return False


def llm_resident_mb():
    """VRAM currently held by llama-server children."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5).stdout
        total = 0
        for line in out.strip().splitlines():
            pid_s, mem_s = line.split(",")
            try:
                with open(f"/proc/{int(pid_s)}/cmdline", "rb") as f:
                    if b"llama-server" in f.read():
                        total += int(mem_s)
            except (OSError, ValueError):
                continue
        return total
    except Exception:
        return 0


def unload_llm():
    """Hand the card back — VERIFIED. A silently-failed eviction once left
    the lyricist squatting 5 GB: the DiT could stage only 206 MB and one
    take took 9 minutes streaming weights from RAM every step."""
    for attempt in (1, 2):
        try:
            urllib.request.urlopen(LLM_URL + "/unload", timeout=65).read()
        except Exception:
            pass
        for _ in range(10):
            if llm_resident_mb() < 500:
                return True
            time.sleep(2)
    log("LLM eviction FAILED — generations will crawl until ttl clears it")
    return False


def listener_waiting():
    """The deck touches this file whenever /next finds an empty spool."""
    try:
        ts = int(open(f"{BASE}/radio/LISTENER_WAITING").read().strip())
        return time.time() - ts < 120
    except (OSError, ValueError):
        return False


def best_accepting_vein(cards, meta_path):
    """The vein most likely to land a take, judged by its own record."""
    counts = {v: 0 for v in cards}
    try:
        with open(meta_path) as f:
            for line in f:
                try:
                    m = json.loads(line)
                except Exception:
                    continue
                if not m.get("event") and m.get("vein") in counts:
                    counts[m["vein"]] += 1
    except OSError:
        pass
    # lyric veins need an LLM residency — too slow for an emergency
    quick = {v: c for v, c in counts.items()
             if "lyrics" not in cards[v].get("vocals", "").lower()}
    pool = quick or counts
    return max(pool, key=pool.get)


def fallback_caption(vein, card):
    """The seed-caption path from sample_caption, without touching the LLM."""
    env = card["envelope"]
    bpm = random.randint(int(env["bpm"][0]), int(env["bpm"][1]))
    lo, hi = env["length_s"]
    target_s = random.randint(int(lo), int(hi))
    want = load_target_len()
    if want:
        target_s = int(max(LEN_MIN, random.uniform(want * 0.85, want * 1.15)))
    # round, do not truncate: // 60 turned a 179s target into "about 2
    # minutes", i.e. a request for 59s less than intended, before the model
    # had done anything. Up to a minute was being lost in the phrasing.
    mins = max(1, round(target_s / 60))
    seed_caption = card["caption_seed"].replace(
        "Global Metadata: ",
        f"Global Metadata: An approximately {mins}-minute piece with a "
        "complete multi-section arc — intro, themes, development, peak, "
        "reprise — before its outro. ", 1)
    # The seed states the vein's dominant arc, twice. Without an LLM this is
    # the ONLY caption path, so leaving it alone would mean every take of
    # every vein has the same shape for anyone running without one. Swap the
    # baked phrase for this take's roll. Hand-tuned seeds do not use these
    # exact phrases, so they are left untouched by design.
    arc_shape, arc = sample_arc(env)
    if arc:
        for phrase in ARC_PHRASES.values():
            if phrase != arc and phrase in seed_caption:
                seed_caption = seed_caption.replace(phrase, arc)
                break
    if arc_shape == "builds_to_end":
        # the seed's own Global Metadata promises an outro; say plainly that
        # this take does not have one, or the taper wins here too
        seed_caption += (" The final section is the loudest point and does "
                         "not fade.")
    # without an LLM this seed is the only caption there is, so the technique
    # roll has to reach it here or those installs get none at all
    seed_caption = seed_caption.rstrip()
    if seed_caption.endswith("."):
        seed_caption = seed_caption[:-1]
    seed_caption += f", built around {random.choice(TECHNIQUES)}."
    return seed_caption, bpm, target_s, arc_shape


def load_bundle_queue(slug):
    """Bundles are a buffered commodity like takes: pre-written to disk so
    an LLM residency almost never blocks production. Survives restarts."""
    mine, others = [], []
    if os.path.exists(BUNDLE_FILE):
        with open(BUNDLE_FILE) as f:
            for line in f:
                try:
                    b = json.loads(line)
                except Exception:
                    continue
                (mine if b.get("station") == slug else others).append(b)
    return mine, others


def save_bundle_queue(slug, mine, others):
    tmp = BUNDLE_FILE + ".tmp"
    with open(tmp, "w") as f:
        for b in others + mine:
            f.write(json.dumps(b) + "\n")
    os.replace(tmp, BUNDLE_FILE)


# recent verdicts per vein, with detail — a vein that keeps losing does NOT
# sit out (a radio must never lose a genre): its essence card gets REWRITTEN
# by the LLM at the next residency, evidence in hand.
_recent = {}
_needs_rewrite = set()
_rewrites_today = {}
FAIL_WINDOW = 8
FAIL_MIN_ATTEMPTS = 6
REWRITES_PER_DAY = 3
REWRITE_RETRY_S = 2 * 3600


def record_verdict(vein, accepted, dur=None, score=None, thr=None):
    h = _recent.setdefault(vein, [])
    h.append({"ok": accepted, "dur": dur, "score": score, "thr": thr})
    del h[:-FAIL_WINDOW]
    oks = sum(1 for x in h if x["ok"])
    if len(h) >= FAIL_MIN_ATTEMPTS and oks / len(h) < 0.25:
        _needs_rewrite.add(vein)
        log(f"vein {vein} failing ({oks}/{len(h)} accepts) — card rewrite "
            "queued for next LLM residency")


def failure_summary(vein):
    h = _recent.get(vein, [])
    parts = []
    for x in h:
        if x["ok"]:
            parts.append(f"ACCEPT {x['dur']:.0f}s")
        elif x["score"] is None:
            parts.append(f"STUB {x['dur']:.0f}s (too short)")
        else:
            parts.append(f"MISS {x['dur']:.0f}s score {x['score']:.2f} "
                         f"(needed {x['thr']:.2f})")
    return "; ".join(parts)


def corpus_evidence(analysis, vein):
    """What this vein's real corpus tracks sound like, per the captioner."""
    try:
        with open(f"{analysis}/veins.json") as f:
            central = json.load(f)["veins"][vein]["central_tracks"][:6]
        caps = {}
        with open(f"{analysis}/captions.jsonl") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    caps[r["path"]] = r["caption"]
                except Exception:
                    continue
        return [caps[t][:450] for t in central if t in caps][:2]
    except Exception:
        return []


def rewrite_card(vein, card, analysis, cards_path):
    """The self-repair loop: hand the LLM the failing card, the failure
    pattern, the corpus ground truth, and the physics lessons — get back a
    better card. The old card is preserved in history, always revertible."""
    day = int(time.time() // 86400)
    n, last, last_day = _rewrites_today.get(vein, (0, 0, day))
    if last_day != day:
        n = 0
    if n >= REWRITES_PER_DAY and time.time() - last < REWRITE_RETRY_S:
        return False
    evidence = corpus_evidence(analysis, vein)
    prompt = (
        "A music-generation vein keeps failing and you must rewrite its "
        "essence card. Return STRICT JSON with exactly these keys: "
        '{"essence": str, "caption_seed": str, "fixed_core": [str, ...]}\n\n'
        f"CURRENT CARD:\n{json.dumps({k: card[k] for k in ('essence', 'caption_seed', 'fixed_core')}, indent=1)}\n\n"
        f"RECENT RESULTS: {failure_summary(vein)}\n\n"
        "KNOWN PHYSICS of the generator: track length follows ARRANGEMENT "
        "RICHNESS — captions with many concrete, named sections produce "
        "full-length takes; sparse ones produce 15-30s stubs. State the "
        "duration early in Global Metadata. Genre anchors must be explicit "
        "(the model drifts genre on weak language). caption_seed must keep "
        "the three-section format: Global Metadata / Vocal Details / "
        "Arrangement.\n"
        + (("\nWHAT THE REAL CORPUS TRACKS SOUND LIKE (ground truth — the "
            "card must chase THIS):\n" + "\n---\n".join(evidence) + "\n")
           if evidence else "")
        + "\nSTUBS mean: make the Arrangement longer, denser, more "
          "sequential. MISSES mean: the essence has drifted from the ground "
          "truth — pull it back. Output ONLY the JSON."
    )
    text = llm_chat(prompt, temperature=0.7, max_tokens=1200)
    if not text:
        return False
    try:
        m = re.search(r"\{.*\}", text, re.S)
        new = json.loads(m.group(0))
        assert isinstance(new["essence"], str) and new["essence"]
        assert all(h in new["caption_seed"] for h in
                   ("Global Metadata:", "Vocal Details:", "Arrangement:"))
        assert isinstance(new["fixed_core"], list)
    except Exception as e:
        log(f"card rewrite for {vein} unparseable ({e!r:.60}) — kept old card")
        return False
    with open(f"{analysis}/essence_cards_history.jsonl", "a") as f:
        f.write(json.dumps({"ts": int(time.time()), "vein": vein,
                            "old": {k: card[k] for k in
                                    ("essence", "caption_seed", "fixed_core")},
                            "new": new,
                            "reason": failure_summary(vein)}) + "\n")
    card.update(new)
    with open(cards_path) as f:
        doc = json.load(f)
    doc["veins"][vein].update(new)
    tmp = cards_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(doc, f, indent=1)
    os.replace(tmp, cards_path)
    _rewrites_today[vein] = (n + 1, time.time(), day)
    _recent[vein] = []
    _relief.pop(vein, None)
    _cooldown.pop(vein, None)
    log(f"REWROTE card for vein {vein} (rewrite {n + 1} today) — "
        "fresh slate, no genre sits out")
    return True


def refill_bundles(cards, per, per_n, count, weights_path, p):
    """One LLM residency, several bundles: free the generator, write
    CAPTION_BATCH caption/lyric sets across the neediest veins, unload.
    Failing veins get their cards REWRITTEN first, in the same residency —
    the LLM is already resident, so repair costs no extra swap. The lyric
    cap is enforced pick-by-pick against the simulated spool."""
    sim = dict(per)
    sim_n = dict(per_n)
    n = count
    lv = lyric_veins(cards)
    freed = free_music3()
    # A batch of seed captions is not worth banking. Writing CAPTION_BATCH of
    # them while the card is busy fills a 20-deep queue with the same static
    # card text, and the daemon then spends the next ~40 takes on it — long
    # after the LLM became reachable again. Measured: 20 of 20 recent takes
    # opened with the seed's boilerplate. Write just enough to keep the radio
    # moving, and let the queue refill properly once the card frees.
    batch = CAPTION_BATCH if freed else 2
    picks = []
    for _ in range(batch):
        excl = cap_excludes(cards, sim_n, n)
        v = pick_vein(cards, sim, weights_path, excl)
        # a bundle composes TWICE (uses=2) — the cap simulation must count
        # both takes, or reuse smuggles lyric share past the cap
        sim[v] = sim.get(v, 0.0) + 400.0
        sim_n[v] = sim_n.get(v, 0) + 2
        n += 2
        picks.append(v)
    if freed:
        while _needs_rewrite:
            v = _needs_rewrite.pop()
            if v in cards:
                rewrite_card(v, cards[v], p["analysis"],
                             os.path.join(p["analysis"], "essence_cards.json"))
    out = []
    for v in picks:
        card = cards[v]
        if freed:
            caption, bpm, target_s, arc_shape = sample_caption(v, card)
        else:
            # unfreed card: never wake the LLM into a VRAM squeeze —
            # seed captions keep the radio fed until the card clears
            caption, bpm, target_s, arc_shape = fallback_caption(v, card)
        lyr = ((sample_lyrics(card) if freed else "")
               if v in lv else structural_tags(target_s, arc=arc_shape))
        # uses=2: one caption composes two different songs (the encoder seed
        # is the composer), halving residency demand at zero quality cost
        out.append({"vein": v, "caption": caption, "lyrics": lyr,
                    "bpm": bpm, "target_s": target_s, "uses": 2,
                    "arc": arc_shape,
                    # so the queue can tell a written caption from a seed
                    "seed": not freed})
    if freed:
        unload_llm()
    return out


CAPTION_SCHEMA_EXAMPLE = (
    "Global Metadata: Instrumental lo-fi hip-hop, 80 BPM, warm and mellow, "
    "late-night headphones listening.\n\n"
    "Vocal Details: No vocals, fully instrumental.\n\n"
    "Arrangement: Dusty boom-bap drums, soft kick, round sub bass, warm "
    "Rhodes piano chords, vinyl crackle texture, sparse jazzy guitar fills. "
    "Intro: solo Rhodes with crackle, drums enter. Outro: fade to crackle."
)


# Must agree with analysis/import_pipeline.ARC_PHRASES and cluster.ARC_SHAPES.
ARC_PHRASES = {
    "mid_peak": "builds to a mid peak and lands soft",
    "builds_to_end": "builds steadily and peaks at the end",
    "front_loaded": "starts strong and eases off",
}


def sample_arc(env):
    """Roll this take's energy shape from the vein's own distribution.

    The vein's mean arc is not a shape any particular track has — averaging
    80+ tracks cancels their differences and leaves the universal "intros are
    quieter", so every vein reads as mid-peak. Sending that one phrase every
    time made generation strictly narrower than the library it models: a
    corpus measured at 62% mid-peak / 25% builds-to-end / 13% front-loaded
    was producing 100% mid-peak, so a quarter of its dynamic range was
    unreachable. Rolling per take, like bpm and key, restores the spread.

    Cards written before this existed — and hand-tuned cards, whose arc
    phrases are better than anything generated here — have no distribution,
    and keep their single stated arc.

    MEASURED CAVEAT: this steers the caption, and the caption does not
    reliably steer the audio. Across 20 takes, the requested shape arrived
    40% of the time overall and 0 of 5 times for builds-to-end, because takes
    end at a median 46% of their planned length — the back half of the arc,
    where the shape is decided, never gets generated. Worth keeping (it costs
    nothing and does diversify the captions) but do not read the request as
    a guarantee about the audio.
    """
    shapes = (env or {}).get("energy_shapes")
    stated = (env or {}).get("energy_arc", "")
    if not shapes:
        return (env or {}).get("energy_arc_shape", ""), stated
    labels = [s for s in shapes if shapes[s] > 0]
    if not labels:
        return (env or {}).get("energy_arc_shape", ""), stated
    pick = random.choices(labels, weights=[shapes[s] for s in labels])[0]
    # A hand-written arc describes this vein's usual shape, and does it far
    # better than the three generic phrases here ("verse restraint → chorus
    # lift → bridge dip → final chorus peak"). Keep it for the shape it
    # describes — the most common one — and let the generic phrases cover
    # the shapes the card never mentioned.
    if stated and stated not in ARC_PHRASES.values():
        # Which shape the hand-written phrase actually describes. Defaults to
        # the vein's most common one, which is usually right — but not
        # always: "verse restraint, chorus lift, bridge dip, final chorus
        # peak" is a build to the end, in a vein that is mostly mid-peak.
        # Guessing wrong sends the majority of takes the wrong shape, so a
        # card may say outright with energy_arc_shape.
        if pick == (env.get("energy_arc_shape")
                    or max(shapes, key=shapes.get)):
            return pick, stated
    return pick, ARC_PHRASES.get(pick, stated)


# The negative lookahead does the work a trailing \b cannot: \b sits between
# two word characters, so "F#" followed by "/" would fail it and backtrack to
# a bare "F", silently transposing the key. Requiring "not followed by a
# letter" keeps the accidental and still refuses to read the G in "Great".
KEY_RE = re.compile(r"\b([A-G][#b♯♭]?)(?![A-Za-z])(?:\s+(major|minor))?")

# Compositional technique, rolled per take.
#
# Everything else the caption varies is surface: which instrument, how fast,
# how long, how bright. Nothing asked for anything to HAPPEN in the music, so
# takes came out as the same piece in different clothes. These are phrased as
# events a text-to-music model can act on — no theory vocabulary, each one
# something you could point at in a waveform.
TECHNIQUES = [
    "a call-and-response between the lead and a second, answering voice",
    "one repeating figure held underneath while the harmony moves around it",
    "a key change lifting the final section",
    "a half-time feel for one section, then back to the original pulse",
    "everything dropping away to a single element before the last section",
    "a countermelody entering in the second half to answer the main theme",
    "the opening theme returning at the end played by a different instrument",
    "the main figure displaced off the beat, landing late against the pulse",
    "layers stacked one at a time, then stripped back to where it started",
    "a tempo pull-back easing into the closing section",
    "syncopated accents cutting across a steady pulse",
    "a solo passage that develops the main motif instead of repeating it",
    "a shift from bright harmony to a darker mode partway through",
    "call-and-answer traded between the low and high registers",
    "a build made from filtering and texture alone, with no new instruments",
    "the melody returning in longer, slower notes near the end",
    "a breakdown to bare percussion before the final section",
    "two contrasting themes alternating, the second gaining ground each time",
]


def sample_key(env):
    """One key from the vein's own keys, in either card dialect.

    Auto-generated cards write a plain list ("C major, G major, D minor");
    hand-tuned ones write prose ("major-leaning (C, G, D major; D minor for
    the wistful ones)"). The old parser only handled the parenthesised form,
    so every auto card — every new install — silently produced no key at all
    and the prompt fell back to "your choice". It also split on commas after
    taking the parenthetical, which dropped the mode and left bare "C"."""
    raw = (env or {}).get("keys", "") or ""
    if "(" in raw:  # prose form: the parenthetical holds the actual keys
        raw = raw.split("(", 1)[1].rsplit(")", 1)[0]
    raw = raw.split(";")[0]
    # Pull key-like tokens out rather than splitting on punctuation: real
    # cards write "C, G, D major", "G/D/A minor" and "G major for the warm
    # resolves", and a comma split turns the last two into a single unusable
    # "key". The note letter stays case-sensitive on purpose — a lowercase
    # "a" in surrounding prose is a word, not a key.
    found = KEY_RE.findall(raw)
    if not found:
        return ""
    # A group states its mode once, at the end: "C, G, D major" is three
    # major keys, and "G/D/A minor, G major" is three minor keys and one
    # major. So a bare note takes the next mode stated after it, not the
    # last one in the string — reading backwards would turn that G/D/A
    # minor group major.
    keys, pending = [], []
    for note, mode in found:
        pending.append(note)
        if mode:
            keys += [f"{n} {mode}" for n in pending]
            pending = []
    trailing = next((m for _, m in reversed(found) if m), "")
    keys += [f"{n} {trailing}".strip() for n in pending]
    return random.choice(keys) if keys else ""


def sample_caption(vein, card):
    """Roll concrete constraints here (LLMs randomize poorly), then have
    Gemma write a fresh caption inside them."""
    env = card["envelope"]
    bpm = random.randint(int(env["bpm"][0]), int(env["bpm"][1]))
    key = sample_key(env)
    axes = random.sample(card["mutation_axes"], k=min(2, len(card["mutation_axes"])))
    arc_shape, arc = sample_arc(env)
    technique = random.choice(TECHNIQUES)
    lo, hi = card["envelope"]["length_s"]
    target_s = random.randint(int(lo), int(hi))
    want = load_target_len()
    if want:
        # keep some spread around the listener's choice, so a fixed setting
        # does not make every take exactly the same length
        target_s = int(max(LEN_MIN, random.uniform(want * 0.85, want * 1.15)))

    # round, do not truncate: // 60 turned a 179s target into "about 2
    # minutes", i.e. a request for 59s less than intended, before the model
    # had done anything. Up to a minute was being lost in the phrasing.
    mins = max(1, round(target_s / 60))
    # Precomputed, not spliced into the prompt's concatenation chain: a bare
    # conditional in the middle of implicit string joining is a syntax error
    # waiting to happen, and this line only appears for one arc.
    no_fade = ("THE END IS THE PEAK: the final section is the loudest, "
               "fullest point of the piece. Do not describe a fade, a "
               "wind-down or a gentle close — it finishes at full strength.\n"
               if arc_shape == "builds_to_end" else "")
    prompt = (
        "You write captions for a music generation model. The caption format "
        "has exactly three sections, like this example:\n\n"
        f"{CAPTION_SCHEMA_EXAMPLE}\n\n"
        f"Write ONE new caption for a track in this style vein:\n"
        f"ESSENCE: {card['essence']}\n"
        f"NEVER LOSE: {'; '.join(card['fixed_core'])}\n"
        f"VOCALS: {card['vocals']}\n"
        f"CONSTRAINTS: {bpm} BPM. Key feel: {key or 'your choice within the vein'}. "
        f"Spectral character: {env['spectral']}. Energy shape: {arc}.\n"
        f"TARGET LENGTH: about {mins} minutes — state the approximate duration "
        "in Global Metadata (e.g. 'a four-minute piece'), and write the "
        "Arrangement as at least five named sections in sequence (intro, "
        "first theme, development or second theme, peak or bridge, reprise, "
        "outro), each with concrete content, so the full duration is "
        "accounted for. The model ends songs early when the arc is thin — "
        "give it a complete journey.\n"
        f"VARY THESE (be specific, commit to concrete choices): {'; '.join(axes)}.\n"
        # its own line, not folded into VARY THESE: this is the one thing the
        # piece should DO, and it loses to instrument choice when they compete
        f"TECHNIQUE — build the arrangement around this and name where it "
        f"happens: {technique}.\n"
        f"{no_fade}"
        "Do not mention any real artist, band, game, or song name. "
        "Output ONLY the caption text, three sections, no commentary."
    )
    for _ in range(2):
        text = llm_chat(prompt, temperature=1.0, max_tokens=700)
        if text and all(h in text for h in
                        ("Global Metadata:", "Vocal Details:", "Arrangement:")):
            return text, bpm, target_s, arc_shape
    log("caption fallback: using card seed")
    # seeds predate the length discipline. Inject the duration INTO Global
    # Metadata — trailing it after the outro description reads as post-song
    # text and the encoder still ends early (measured: 16s takes).
    seed_caption = card["caption_seed"].replace(
        "Global Metadata: ",
        f"Global Metadata: An approximately {mins}-minute piece with a "
        "complete multi-section arc — intro, themes, development, peak, "
        "reprise — before its outro. ", 1)
    return seed_caption, bpm, target_s, arc_shape


def sample_lyrics(card):
    prompt = (
        f"Write original lyrics for a song in this style: {card['essence']}\n"
        "Voice: earnest, concrete everyday imagery, zero irony. Invent a "
        "fresh specific mundane moment as the theme.\n"
        "Imagery world: 90s suburban-indoor — apartments, parking lots, "
        "payphones, TV static, mixtapes, late buses, fluorescent stores. "
        "NO rural/Americana markers (porches, gardens, dirt roads, trucks, "
        "whiskey) — the generator hears those as country and twangs the "
        "whole arrangement.\n"
        "Structure with section tags: [intro] [verse] [chorus] [verse] "
        "[chorus] [instrumental] [bridge] [chorus] [outro] — the tags are "
        "executable song structure, so use all of them. Under 260 words.\n"
        "Parentheses are allowed ONLY for sung backing words or ad-libs — "
        "never instrument, mood, or production directions.\n"
        "Output ONLY the tagged lyrics."
    )
    text = llm_chat(prompt, temperature=1.0, max_tokens=800)
    if not text or "[" not in text:
        return ""
    # strip any parenthetical that reads as direction, not singing
    import re
    def keep(m):
        inner = m.group(1)
        words = inner.split()
        bad = ("guitar", "drum", "beat", "synth", "piano", "fade", "solo",
               "music", "instrumental", "strum", "riff", "bass", "tempo")
        return "" if (len(words) > 6 or any(b in inner.lower() for b in bad)) \
            else m.group(0)
    return re.sub(r"\(([^)]*)\)", keep, text).strip()


# ------------------------------------------------------------------- critic

_clap = None


def clap_embed(path):
    """CPU on purpose: generation owns the GPU, and a few seconds of scoring
    is nothing next to a multi-minute generation."""
    global _clap
    import librosa
    import numpy as np
    import torch
    import torch.nn.functional as F
    if _clap is None:
        from transformers import ClapModel, ClapProcessor
        m = ClapModel.from_pretrained("laion/clap-htsat-unfused").eval()
        p = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
        _clap = (m, p)
    m, p = _clap
    y, sr = librosa.load(path, sr=48000, mono=True)
    n = max(1, int(len(y) / (sr * 10)))
    wins = [y[i * sr * 10:(i + 1) * sr * 10] for i in range(n)]
    wins = [w for w in wins if len(w) >= sr * 3] or [y[: sr * 10]]
    vecs = []
    for w in wins:
        inp = p(audio=w, sampling_rate=sr, return_tensors="pt")
        with torch.no_grad():
            out = m.get_audio_features(**inp)
            if hasattr(out, "pooler_output"):
                out = out.pooler_output
            vecs.append(out.squeeze(0))
    v = torch.stack(vecs).mean(dim=0)
    return F.normalize(v, dim=-1).numpy()


def load_critic(analysis):
    """Per-vein acceptance thresholds, self-calibrated: P10 of each vein's own
    member-to-centroid similarity. A generated track must sit where at least
    the corpus's own fringe sits."""
    import numpy as np
    emb = np.load(f"{analysis}/embeddings.npy")
    with open(f"{analysis}/embeddings_keys.json") as f:
        keys = json.load(f)
    with open(f"{analysis}/veins.json") as f:
        veins = json.load(f)["veins"]
    pos = {k: i for i, k in enumerate(keys)}
    crit = {}
    for label, v in veins.items():
        c = np.array(v["centroid"])
        sims = [float(emb[pos[t]] @ c) for t in v["all_tracks"] if t in pos]
        crit[label] = {"centroid": c,
                       "threshold": float(np.percentile(sims, 10))}
    return crit


# Corpus P10 asks "where does the fringe of my own music sit?" — the right
# anchor for a broad corpus, the wrong one for a tight corpus. A one-artist
# library is so self-similar that its P10 lands around 0.93, and generated
# audio never embeds that close to a corpus centroid, so every take is
# rejected forever and the radio plays nothing. (Relief cannot save it:
# 0.05 of easing against a 0.25 gap, snapped back on every accept.)
#
# So the bar is ALSO ranked against what the vein actually generates — keep
# the better half of its own output. The lower of the two bars wins, floored
# so a vein that generates uniformly badly still cannot bank noise. This can
# only ease the corpus bar, never raise it: where the corpus bar is already
# reachable, the critic behaves exactly as it did before.
CRITIC_WINDOW = 60      # generated scores remembered per vein
CRITIC_MIN_SAMPLES = 4  # scores before the learned bar is trusted
CRITIC_KEEP_FRAC = float(_cfg.get("critic_keep_frac", 0.5))
CRITIC_FLOOR = float(_cfg.get("critic_floor", 0.45))


def load_scores(analysis):
    """Generated-score history per vein, persisted so a restart does not
    replay the calibration window."""
    try:
        with open(f"{analysis}/critic_scores.json") as f:
            raw = json.load(f)
        return {v: [float(x) for x in s][-CRITIC_WINDOW:]
                for v, s in raw.items()}
    except (OSError, ValueError, TypeError, AttributeError):
        return {}


def save_scores(analysis, scores):
    path = f"{analysis}/critic_scores.json"
    try:
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({v: [round(x, 4) for x in s]
                       for v, s in scores.items()}, f)
        os.replace(tmp, path)
    except OSError:
        pass


def record_score(scores, vein, score):
    h = scores.setdefault(vein, [])
    h.append(float(score))
    del h[:-CRITIC_WINDOW]


VERDICT_KEEP = 24


def publish_verdict(analysis, **v):
    """The cutting-room floor, on disk for the deck to draw.

    A rejected take is deleted, so without this the listener sees a radio
    that is plainly working hard and has nothing to show for it — the most
    common 'is it broken?' moment in the whole system."""
    path = f"{analysis}/critic_recent.json"
    try:
        with open(path) as f:
            recent = json.load(f)[-VERDICT_KEEP + 1:]
    except (OSError, ValueError):
        recent = []
    recent.append({**v, "ts": int(time.time())})
    try:
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(recent, f)
        os.replace(tmp, path)
    except OSError:
        pass


BIAS_MAX = 0.15  # how far the listener may move the bar, either way


def bias_path(analysis):
    return f"{analysis}/critic_bias.json"


def load_bias(analysis):
    """The listener's own thumb on the scale, read fresh every take so the
    deck's stricter/looser buttons apply without a restart."""
    try:
        with open(bias_path(analysis)) as f:
            return max(-BIAS_MAX, min(BIAS_MAX, float(json.load(f)["bias"])))
    except (OSError, ValueError, TypeError, KeyError):
        return 0.0


def critic_bar(vein, critic, scores, bias=0.0):
    """The bar this take must clear, and the reason, for the log."""
    import numpy as np
    corpus = critic[vein]["threshold"]
    hist = scores.get(vein, [])
    if CRITIC_KEEP_FRAC <= 0:  # opt out: corpus bar only, as it was
        return corpus + bias, f"corpus {corpus:.3f}"
    if len(hist) < CRITIC_MIN_SAMPLES:
        return corpus + bias, (f"corpus {corpus:.3f}, calibrating "
                               f"{len(hist)}/{CRITIC_MIN_SAMPLES}")
    learned = max(CRITIC_FLOOR,
                  float(np.percentile(hist, 100 * (1.0 - CRITIC_KEEP_FRAC))))
    base, why = ((corpus, f"corpus {corpus:.3f}") if learned >= corpus
                 else (learned, f"learned {learned:.3f} from {len(hist)} "
                                f"takes, corpus {corpus:.3f} out of reach"))
    if bias:
        why += f", you {'raised' if bias > 0 else 'lowered'} it by {abs(bias):.02f}"
    return base + bias, why


# --------------------------------------------------------------- generation

def roll_engine():
    """The generator's own diversity settings, per take.

    Separate AR seed on purpose: the text encoder's seed drives acoustic-token
    sampling and KSampler's drives diffusion noise. They are independent
    stages, and feeding both the same number tied two draws together for no
    reason."""
    return {
        "ar_seed": random.randrange(2 ** 31),
        "ar_cfg": round(random.uniform(*AR_CFG_RANGE), 2),
        "top_k": random.randint(*AR_TOP_K_RANGE),
    }


def build_graph(caption, lyrics, seed, max_duration, steps=None, eng=None):
    eng = eng or {"ar_seed": seed, "ar_cfg": CFG, "top_k": TOP_K}
    return {
        "1": {"class_type": "CLIPLoader", "inputs": {
            "clip_name": "minimax_music3_text_encoder_pruned_int8_convrot.safetensors",
            "type": "minimax", "device": "default"}},
        "2": {"class_type": "UNETLoader", "inputs": {
            "unet_name": DIT_MODEL,
            "weight_dtype": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {
            "vae_name": "minimax_music3_dav.safetensors"}},
        "4": {"class_type": "MiniMaxMusic3TextEncode", "inputs": {
            "clip": ["1", 0], "caption": caption, "lyrics": lyrics,
            "seed": eng["ar_seed"], "max_duration": float(max_duration),
            "cfg_scale": eng["ar_cfg"], "top_k": eng["top_k"]}},
        "5": {"class_type": "ConditioningZeroOut",
              "inputs": {"conditioning": ["4", 0]}},
        "6": {"class_type": "EmptyMiniMaxMusic3LatentAudio", "inputs": {
            "seconds": ["4", 1], "batch_size": 1}},
        "7": {"class_type": "KSampler", "inputs": {
            "model": ["2", 0], "positive": ["4", 0], "negative": ["5", 0],
            "latent_image": ["6", 0], "seed": seed,
            "steps": steps or STEPS,
            "cfg": CFG, "sampler_name": "euler", "scheduler": "simple",
            "denoise": 1.0}},
        "8": {"class_type": "VAEDecodeAudioTiled", "inputs": {
            "samples": ["7", 0], "vae": ["3", 0],
            "tile_size": 1536, "overlap": 64}},
        "9": {"class_type": "SaveAudioMP3", "inputs": {
            "audio": ["8", 0], "filename_prefix": "radio_tank/take",
            "quality": "V0"}},
    }


def generate(caption, lyrics, seed, max_duration, steps=None, eng=None):
    pid = http_json(M3 + "/prompt",
                    {"prompt": build_graph(caption, lyrics, seed,
                                           max_duration,
                                           steps, eng)})["prompt_id"]
    t0 = time.time()
    while time.time() - t0 < 1200:
        if _stop:
            return None, "stopped"
        time.sleep(10)
        try:
            h = http_json(M3 + f"/history/{pid}", timeout=10)
        except Exception:
            continue
        if pid not in h:
            # A restarted ComfyUI loses its queue and its history, so this id
            # will never appear and polling it burns the full timeout — 20
            # minutes of dead air for a server bounce. If the id is in neither
            # history nor the queue, and the server is up enough to answer,
            # the work is gone: say so and let the caller re-cut it.
            try:
                q = http_json(M3 + "/queue", timeout=5)
                queued = any(pid in str(item) for item in
                             (q.get("queue_running") or [])
                             + (q.get("queue_pending") or []))
            except Exception:
                queued = True  # cannot tell: keep waiting rather than guess
            if not queued and time.time() - t0 > 30:
                return None, "prompt lost (server restarted?)"
            continue
        entry = h[pid]
        status = entry.get("status", {}).get("status_str")
        if status != "success":
            return None, status or "failed"
        for o in entry.get("outputs", {}).values():
            for a in o.get("audio", []):
                return os.path.join(COMFY_OUT, a.get("subfolder", ""),
                                    a["filename"]), "success"
        return None, "no-output"
    return None, "timeout"


# --------------------------------------------------------------------- tank

def tank_seconds(meta_path):
    """meta.jsonl is an event log: track lines add to the tank, consumption
    events (appended by the radio deck when it serves a track) remove."""
    recs, consumed = {}, set()
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            for line in f:
                try:
                    m = json.loads(line)
                except Exception:
                    continue
                if m.get("event") == "consumed":
                    consumed.add(m["id"])
                elif "id" in m:
                    recs[m["id"]] = m
    per = {}
    per_n = {}
    total = 0.0
    count = 0
    for rid, m in recs.items():
        if rid in consumed or m.get("consumed"):
            continue
        per[m["vein"]] = per.get(m["vein"], 0.0) + m["duration_s"]
        per_n[m["vein"]] = per_n.get(m["vein"], 0) + 1
        total += m["duration_s"]
        count += 1
    return total, per, count, per_n


def duration_of(path):
    import av
    with av.open(path) as c:
        return round(float(c.duration) / av.time_base, 1)


CONSUMED_GRACE_S = 48 * 3600


def janitor(meta_path, tank_dir, keepers_dir):
    """Consumed, unkept takes leave the tank after a grace window.

    Keep = copied to keepers/ by the deck, so deleting the tank copy loses
    nothing Dean chose to hold. The grace window keeps 'wait, replay that
    one' possible for two days."""
    recs, consumed = {}, set()
    if not os.path.exists(meta_path):
        return
    with open(meta_path) as f:
        for line in f:
            try:
                m = json.loads(line)
            except Exception:
                continue
            if m.get("event") == "consumed":
                consumed.add(m["id"])
            elif not m.get("event") and "id" in m:
                recs[m["id"]] = m
    kept = set(os.listdir(keepers_dir)) if os.path.isdir(keepers_dir) else set()
    now = time.time()
    for rid in consumed:
        m = recs.get(rid)
        if not m or m["file"] in kept:
            continue
        path = os.path.join(tank_dir, m["file"])
        try:
            if os.path.exists(path) and now - os.path.getmtime(path) > CONSUMED_GRACE_S:
                os.unlink(path)
                log(f"janitor: removed consumed take {m['file']}")
        except OSError:
            pass


def effective_weights(cards, weights_path):
    """Card weights scaled by the deck's feedback multipliers, renormalized.
    keep -> vein grows, dislike -> shrinks; the file is written by the UI."""
    mult = {}
    if os.path.exists(weights_path):
        try:
            with open(weights_path) as f:
                mult = json.load(f)
        except Exception:
            pass
    w = {l: c["weight"] * float(mult.get(l, 1.0)) for l, c in cards.items()}
    s = sum(w.values()) or 1.0
    return {l: v / s for l, v in w.items()}


_cooldown = {}  # vein -> unix ts until which it sits out (recent rejects)
COOLDOWN_S = 600


def cool_vein(vein):
    _cooldown[vein] = time.time() + COOLDOWN_S


def pick_vein(cards, per, weights_path, exclude=frozenset()):
    """Most-underfilled vein by RELATIVE deficit, so early fills interleave
    across veins instead of grinding the largest one to target first.
    Recently-rejected veins sit out a cooldown; `exclude` (the lyric cap) is
    hard — an excluded vein is not picked even if it is the only one cold."""
    w = effective_weights(cards, weights_path)
    now = time.time()
    deficits = {}
    for label in cards:
        if label in exclude and len(exclude) < len(cards):
            continue
        target = max(1.0, w[label] * TANK_TARGET_S)
        deficits[label] = max(0.0, target - per.get(label, 0.0)) / target
    warm = {l: d for l, d in deficits.items() if _cooldown.get(l, 0) < now}
    pool = warm or deficits  # all cooling -> ignore cooldowns
    return max(pool, key=pool.get)


def load_station(slug, cache):
    """Cards + critic for a station, cached; None if it is not captured yet."""
    if slug in cache:
        return cache[slug]
    p = stations.paths(slug)
    try:
        with open(os.path.join(p["analysis"], "essence_cards.json")) as f:
            cards = json.load(f)["veins"]
        critic = load_critic(p["analysis"])
    except FileNotFoundError:
        return None
    for d in (p["tank"], p["keepers"]):
        os.makedirs(d, exist_ok=True)
    cache[slug] = (p, cards, critic, load_scores(p["analysis"]))
    return cache[slug]


def main():
    signal.signal(signal.SIGTERM, _sigterm)
    stations.ensure_registry()
    cache = {}
    last_slug = None
    bundles, other_bundles = [], []
    consec_timeouts = 0

    while not _stop:
        slug = stations.active()
        loaded = load_station(slug, cache)
        if loaded is None:
            log(f"station '{slug}' has no capture yet — holding")
            time.sleep(LOOP_SLEEP)
            continue
        p, cards, critic, scores = loaded
        if slug != last_slug:
            log(f"station: {slug} ({len(cards)} vein(s)), "
                f"target {TANK_TARGET_TRACKS} takes, steps {STEPS}")
            log("critic bars — " + ", ".join(
                f"{v}: {critic_bar(v, critic, scores, load_bias(p['analysis']))[1]}"
                for v in sorted(cards) if v in critic))
            last_slug = slug
            bundles, other_bundles = load_bundle_queue(slug)
            if bundles:
                log(f"{len(bundles)} pre-written bundles on disk")

        # Failure class: a full disk turns every accept into a crash loop.
        # Hold generation loudly; serving the existing spool needs no space.
        if shutil.disk_usage(BASE).free < 2 * 2 ** 30:
            log("DISK nearly full (<2 GiB) — holding generation")
            time.sleep(300)
            continue

        janitor(p["meta"], p["tank"], p["keepers"])
        total, per, count, per_n = tank_seconds(p["meta"])
        if count >= TANK_TARGET_TRACKS:
            # idle window: the card is free and nobody needs takes — bank
            # bundles instead, so future residencies never block production
            if len(bundles) < BUNDLE_QUEUE_TARGET and not hold_reason():
                log(f"spool full — pre-writing bundles "
                    f"({len(bundles)}/{BUNDLE_QUEUE_TARGET} banked)")
                bundles += refill_bundles(cards, per, per_n, count,
                                          p["weights"], p)
                for b in bundles:
                    b["station"] = slug
                save_bundle_queue(slug, bundles, other_bundles)
                continue
            log(f"spool full ({count} takes banked, {total/60:.0f} min)")
            if ONESHOT:
                return
            time.sleep(LOOP_SLEEP)
            continue

        reason = hold_reason()
        if reason:
            log(f"holding: {reason}")
            time.sleep(LOOP_SLEEP)
            continue

        if not bundles:
            if total < 60 and listener_waiting():
                # someone is sitting at a silent deck: skip the batch
                # residency, cut ONE fast seed-caption take from the vein
                # that historically lands, and get audio moving
                vein = best_accepting_vein(cards, p["meta"])
                card = cards[vein]
                caption, bpm, target_s, arc_shape = fallback_caption(vein, card)
                log(f"EMERGENCY take for waiting listener: {card['name']}")
                bundles = [{"vein": vein, "caption": caption,
                            "lyrics": structural_tags(target_s,
                                                      arc=arc_shape),
                            "bpm": bpm, "target_s": target_s,
                            "arc": arc_shape, "station": slug}]
            else:
                log(f"writing {CAPTION_BATCH} bundles (LLM residency)")
                bundles = refill_bundles(cards, per, per_n, count,
                                         p["weights"], p)
                for b in bundles:
                    b["station"] = slug
                save_bundle_queue(slug, bundles, other_bundles)
        # generation-time cap gate — the hard guarantee, whatever the queue
        # holds: lyric bundles rotate to the back while the live spool is at
        # or over the cap, and if EVERY queued bundle is capped, write fresh
        # instrumental ones instead of waiting
        lv_now = lyric_veins(cards)
        picked = None
        for _ in range(len(bundles)):
            cand = bundles.pop(0)
            sung = sum(per_n.get(v, 0) for v in lv_now)
            if (cand["vein"] in lv_now and count > 0
                    and sung / count >= LYRIC_CAP):
                bundles.append(cand)
                continue
            picked = cand
            break
        if picked is None:
            log("all queued bundles lyric-capped — writing instrumentals")
            fresh = refill_bundles(cards, per, per_n, count, p["weights"], p)
            for fb in fresh:
                fb["station"] = slug
            bundles = fresh + bundles
            save_bundle_queue(slug, bundles, other_bundles)
            continue
        b = picked
        b["uses"] = b.get("uses", 1) - 1
        if b["uses"] > 0:
            bundles.append(dict(b))  # back of the queue for its second song
        save_bundle_queue(slug, bundles, other_bundles)
        vein = b["vein"]
        caption, lyrics = b["caption"], b["lyrics"]
        bpm, target_s = b["bpm"], b["target_s"]
        if vein not in cards:
            continue  # cards changed since this bundle was written
        card = cards[vein]
        if TEST_SHORT:
            target_s = 45
        seed = random.randrange(2 ** 31)
        eng = roll_engine()
        lyr_mode = ("words" if vein in lyric_veins(cards)
                    else "tags" if lyrics else "none")
        # First takes into an empty spool are the showcase pair — full
        # quality, always: an empty queue is the demo moment (Dean's rule).
        # Catch-up steps apply only between the wow pair and a healthy spool.
        # Always full quality. The catch-up lever dropped a starved spool to
        # STARVED_STEPS to refill faster, but generation is far enough below
        # realtime that trading quality for it never bought the radio much —
        # and every take it cheapened was one heard while the spool was thin,
        # i.e. exactly when the listener is most likely to be waiting on it.
        steps_wanted = load_steps()
        steps_now = _round_half_up(steps_wanted)
        mode = (f" steps={steps_now}" if steps_wanted == steps_now
                else f" steps={steps_now} (slider {steps_wanted:g})")
        log(f"generating: vein={card['name']} bpm={bpm} "
            f"len<={target_s}s seed={seed} lyrics={lyr_mode}"
            f" top_k={eng['top_k']} arcfg={eng['ar_cfg']}{mode}")

        # final gate right before taking the card
        if hold_reason():
            continue
        if llm_resident_mb() > 1000:
            log("lyricist still on card — evicting before generation")
            unload_llm()
        # max_duration is a cap the encoder undershoots — never let it bind;
        # length is driven by the caption's stated duration and arc instead
        t_gen = time.time()
        path, status = generate(caption, lyrics, seed, 300, steps_now, eng)
        gen_s = round(time.time() - t_gen, 1)
        if not path:
            # Failure class: a hung ComfyUI holds queue_running forever and
            # every cycle burns a 20-minute timeout — slow-motion freeze.
            # Two consecutive timeouts = bounce the music server.
            if status == "timeout":
                consec_timeouts += 1
                if consec_timeouts >= 2:
                    log("two consecutive generation timeouts — "
                        "bouncing comfyui-music3 to self-heal")
                    subprocess.run(["systemctl", "--user", "restart",
                                    "comfyui-music3.service"], timeout=60)
                    consec_timeouts = 0
                    time.sleep(45)
                    continue
            log(f"generation {status}; backoff {BACKOFF}s")
            time.sleep(BACKOFF)
            continue
        consec_timeouts = 0

        try:
            dur = duration_of(path)
            # Absolute stub floor, NOT target-coupled: the model's natural takes
            # run ~60-135s however rich the caption; a floor pinned to the
            # aspirational target rejects the whole distribution and starves the
            # radio. Targets still pull length upward via the caption text.
            if dur < MIN_TAKE_S:
                os.unlink(path)
                cool_vein(vein)
                record_verdict(vein, False, dur=dur)
                publish_verdict(p["analysis"], vein=vein,
                                vein_name=card["name"], ok=False,
                                score=None, thr=None, dur=round(dur),
                                why=f"stub — under the {MIN_TAKE_S}s floor")
                log(f"REJECT-stub {card['name']} {dur:.0f}s "
                    f"(< {MIN_TAKE_S}s) — vein cools {COOLDOWN_S}s")
                if ONESHOT:
                    return
                continue

            v = clap_embed(path)
            score = float(v @ critic[vein]["centroid"])
            bias = load_bias(p["analysis"])
            thr_base, why = critic_bar(vein, critic, scores, bias)
            thr = thr_base - _relief.get(vein, 0.0)
            # every scored take teaches the bar, accepted or not — a fixed
            # quantile of the vein's own output is what keeps it stable
            record_score(scores, vein, score)
            save_scores(p["analysis"], scores)
            if score >= thr:
                track_id = f"{int(time.time())}_{seed}"
                dest = os.path.join(p["tank"], f"v{vein}__{track_id}.mp3")
                os.replace(path, dest)
                with open(p["meta"], "a") as f:
                    f.write(json.dumps({
                        "id": track_id, "vein": vein, "vein_name": card["name"],
                        "file": os.path.basename(dest), "duration_s": dur,
                        "score": round(score, 3), "threshold": round(thr, 3),
                        "relief": round(_relief.get(vein, 0.0), 3),
                        "bpm": bpm, "seed": seed, "steps": steps_now,
                        # wall seconds this take spent generating, so speed is
                        # a measured ratio rather than one inferred from log
                        # timestamps: duration_s / gen_s is the real x-realtime
                        "gen_s": gen_s,
                        # banked with the take so a listener's keep/skip can
                        # later be read back against the settings that made it
                        "engine": eng,
                        # what shape this take was ASKED for, so the audio can
                        # later be measured against the request rather than
                        # against an aggregate
                        "arc_requested": b.get("arc"),
                        "caption": caption, "lyrics": lyrics,
                        "created": int(time.time()), "consumed": False,
                    }) + "\n")
                record_verdict(vein, True, dur=dur, score=score, thr=thr)
                eased = _relief.pop(vein, 0.0)
                publish_verdict(p["analysis"], vein=vein,
                                vein_name=card["name"], ok=True,
                                score=round(score, 3), thr=round(thr, 3),
                                dur=round(dur), why=why)
                log(f"ACCEPT {card['name']} {dur:.0f}s in {gen_s:.0f}s "
                    f"({dur / max(gen_s, 1e-6):.2f}x) score {score:.3f} "
                    f"(thr {thr:.3f}) -> {os.path.basename(dest)}"
                    + (f" — bar restored to {thr_base:.3f}" if eased else ""))
            else:
                os.unlink(path)
                cool_vein(vein)
                record_verdict(vein, False, dur=dur, score=score, thr=thr)
                _relief[vein] = min(RELIEF_MAX,
                                    _relief.get(vein, 0.0) + RELIEF_STEP)
                publish_verdict(p["analysis"], vein=vein,
                                vein_name=card["name"], ok=False,
                                score=round(score, 3), thr=round(thr, 3),
                                dur=round(dur), why=why,
                                short=round(thr - score, 3))
                log(f"REJECT {card['name']} {dur:.0f}s score {score:.3f} "
                    f"< thr {thr:.3f} [{why}] — bar eases to "
                    f"{thr_base - _relief[vein]:.3f}, vein cools {COOLDOWN_S}s")
        except Exception as e:
            # Failure class: one corrupt output (undecodable mp3, CLAP
            # choke) must cost one take, never the daemon.
            log(f"verdict failed ({e!r:.90}) — discarding take, cooling vein")
            try:
                os.unlink(path)
            except OSError:
                pass
            cool_vein(vein)
            record_verdict(vein, False, dur=0)
        if ONESHOT:
            return


if __name__ == "__main__":
    main()
