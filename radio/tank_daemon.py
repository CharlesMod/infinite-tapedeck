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


def structural_tags(target_s):
    """Tag-only lyric sheet for instrumentals. The lyric vein outperforms
    because section tags are executable structure and structure is length —
    so every vein gets the scaffold, without putting words in its mouth."""
    mins = max(1, int(target_s) // 60)
    body = ["[intro]"]
    for i in range(max(2, mins + 1)):
        body.append("[instrumental]")
        if i == 1:
            body.append("[bridge]")
    body.append("[outro]")
    return "\n\n".join(body)
# Adaptive bar (Dean's design): every critic reject eases that vein's bar a
# little so dead air self-limits; the next accept snaps it back to the
# calibrated level. Relief is tactical, never a redefinition of taste.
RELIEF_STEP = float(_cfg.get("relief_step", 0.008))
RELIEF_MAX = float(_cfg.get("relief_max", 0.05))
_relief = {}                  # vein -> current threshold relief

STEPS = int(_cfg["steps"])
STARVED_STEPS = int(_cfg.get("starved_steps", 20))
STARVED_BELOW_TAKES = int(_cfg.get("starved_below_takes", 5))
DIT_MODEL = _cfg.get("dit_model", "minimax_music3_dit_fp16.safetensors")
CFG = 1.7
TOP_K = 50
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
            if int(pid_s) not in ours:
                total += int(mem_s)
        except ValueError:
            continue
    return total


def hold_reason():
    if os.path.exists(PAUSE_FLAG):
        return "PAUSE file present"
    for sib in SIBLINGS:
        if queue_busy(sib):
            return f"sibling busy: {sib}"
    if queue_busy(M3) is None:
        return "music3 server down"
    if foreign_vram_mb() > FOREIGN_VRAM_MB:
        return f"foreign VRAM > {FOREIGN_VRAM_MB} MB (game?)"
    return None


# ------------------------------------------------------------------- lyrics

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
            }, timeout=300)
            return r["choices"][0]["message"]["content"].strip()
        except Exception as e:
            log(f"llm attempt {attempt} failed: {e!r:.80}")
            time.sleep(10)
    return None


def free_music3():
    """Hand the card over: drop the generator's weights and wait for VRAM."""
    try:
        http_json(M3 + "/free", {"unload_models": True, "free_memory": True},
                  timeout=15)
    except Exception:
        return
    for _ in range(15):
        try:
            used = int(subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5).stdout.strip())
            if used < 5000:
                return
        except Exception:
            return
        time.sleep(2)


def unload_llm():
    """Hand the card back: evict the LLM before the next generation."""
    try:
        urllib.request.urlopen(LLM_URL + "/unload", timeout=65).read()
    except Exception:
        pass  # ttl will get it eventually; generation may just retry


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
    mins = target_s // 60
    seed_caption = card["caption_seed"].replace(
        "Global Metadata: ",
        f"Global Metadata: An approximately {mins}-minute piece with a "
        "complete multi-section arc — intro, themes, development, peak, "
        "reprise — before its outro. ", 1)
    return seed_caption, bpm, target_s


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
    picks = []
    for _ in range(CAPTION_BATCH):
        excl = cap_excludes(cards, sim_n, n)
        v = pick_vein(cards, sim, weights_path, excl)
        sim[v] = sim.get(v, 0.0) + 200.0  # assume a take lands, move on
        sim_n[v] = sim_n.get(v, 0) + 1
        n += 1
        picks.append(v)
    free_music3()
    while _needs_rewrite:
        v = _needs_rewrite.pop()
        if v in cards:
            rewrite_card(v, cards[v], p["analysis"],
                         os.path.join(p["analysis"], "essence_cards.json"))
    out = []
    for v in picks:
        card = cards[v]
        caption, bpm, target_s = sample_caption(v, card)
        lyr = (sample_lyrics(card) if v in lv
               else structural_tags(target_s))
        # uses=2: one caption composes two different songs (the encoder seed
        # is the composer), halving residency demand at zero quality cost
        out.append({"vein": v, "caption": caption, "lyrics": lyr,
                    "bpm": bpm, "target_s": target_s, "uses": 2})
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


def sample_caption(vein, card):
    """Roll concrete constraints here (LLMs randomize poorly), then have
    Gemma write a fresh caption inside them."""
    env = card["envelope"]
    bpm = random.randint(int(env["bpm"][0]), int(env["bpm"][1]))
    key = random.choice(env["keys"].split("(")[-1].rstrip(")").split(";")[0]
                        .split(",")).strip() if "(" in env["keys"] else ""
    axes = random.sample(card["mutation_axes"], k=min(2, len(card["mutation_axes"])))
    lo, hi = card["envelope"]["length_s"]
    target_s = random.randint(int(lo), int(hi))

    mins = target_s // 60
    prompt = (
        "You write captions for a music generation model. The caption format "
        "has exactly three sections, like this example:\n\n"
        f"{CAPTION_SCHEMA_EXAMPLE}\n\n"
        f"Write ONE new caption for a track in this style vein:\n"
        f"ESSENCE: {card['essence']}\n"
        f"NEVER LOSE: {'; '.join(card['fixed_core'])}\n"
        f"VOCALS: {card['vocals']}\n"
        f"CONSTRAINTS: {bpm} BPM. Key feel: {key or 'your choice within the vein'}. "
        f"Spectral character: {env['spectral']}. Energy shape: {env['energy_arc']}.\n"
        f"TARGET LENGTH: about {mins} minutes — state the approximate duration "
        "in Global Metadata (e.g. 'a four-minute piece'), and write the "
        "Arrangement as at least five named sections in sequence (intro, "
        "first theme, development or second theme, peak or bridge, reprise, "
        "outro), each with concrete content, so the full duration is "
        "accounted for. The model ends songs early when the arc is thin — "
        "give it a complete journey.\n"
        f"VARY THESE (be specific, commit to concrete choices): {'; '.join(axes)}.\n"
        "Do not mention any real artist, band, game, or song name. "
        "Output ONLY the caption text, three sections, no commentary."
    )
    for _ in range(2):
        text = llm_chat(prompt, temperature=1.0, max_tokens=700)
        if text and all(h in text for h in
                        ("Global Metadata:", "Vocal Details:", "Arrangement:")):
            return text, bpm, target_s
    log("caption fallback: using card seed")
    # seeds predate the length discipline. Inject the duration INTO Global
    # Metadata — trailing it after the outro description reads as post-song
    # text and the encoder still ends early (measured: 16s takes).
    seed_caption = card["caption_seed"].replace(
        "Global Metadata: ",
        f"Global Metadata: An approximately {mins}-minute piece with a "
        "complete multi-section arc — intro, themes, development, peak, "
        "reprise — before its outro. ", 1)
    return seed_caption, bpm, target_s


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


# --------------------------------------------------------------- generation

def build_graph(caption, lyrics, seed, max_duration, steps=None):
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
            "seed": seed, "max_duration": float(max_duration),
            "cfg_scale": CFG, "top_k": TOP_K}},
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


def generate(caption, lyrics, seed, max_duration, steps=None):
    pid = http_json(M3 + "/prompt",
                    {"prompt": build_graph(caption, lyrics, seed,
                                           max_duration,
                                           steps)})["prompt_id"]
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
    cache[slug] = (p, cards, critic)
    return cache[slug]


def main():
    signal.signal(signal.SIGTERM, _sigterm)
    stations.ensure_registry()
    cache = {}
    last_slug = None
    bundles, other_bundles = [], []

    while not _stop:
        slug = stations.active()
        loaded = load_station(slug, cache)
        if loaded is None:
            log(f"station '{slug}' has no capture yet — holding")
            time.sleep(LOOP_SLEEP)
            continue
        p, cards, critic = loaded
        if slug != last_slug:
            log(f"station: {slug} ({len(cards)} vein(s)), "
                f"target {TANK_TARGET_TRACKS} takes, steps {STEPS}")
            last_slug = slug
            bundles, other_bundles = load_bundle_queue(slug)
            if bundles:
                log(f"{len(bundles)} pre-written bundles on disk")

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
                caption, bpm, target_s = fallback_caption(vein, card)
                log(f"EMERGENCY take for waiting listener: {card['name']}")
                bundles = [{"vein": vein, "caption": caption,
                            "lyrics": structural_tags(target_s),
                            "bpm": bpm, "target_s": target_s,
                            "station": slug}]
            else:
                log(f"writing {CAPTION_BATCH} bundles (LLM residency)")
                bundles = refill_bundles(cards, per, per_n, count,
                                         p["weights"], p)
                for b in bundles:
                    b["station"] = slug
                save_bundle_queue(slug, bundles, other_bundles)
        b = bundles.pop(0)
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
        lyr_mode = ("words" if vein in lyric_veins(cards)
                    else "tags" if lyrics else "none")
        # First takes into an empty spool are the showcase pair — full
        # quality, always: an empty queue is the demo moment (Dean's rule).
        # Catch-up steps apply only between the wow pair and a healthy spool.
        showcase = count < 2
        starved = count < STARVED_BELOW_TAKES
        steps_now = STEPS if (showcase or not starved) else STARVED_STEPS
        mode = (" steps=30 (showcase)" if showcase and starved
                else f" steps={STARVED_STEPS} (catch-up)"
                if steps_now != STEPS else "")
        log(f"generating: vein={card['name']} bpm={bpm} "
            f"len<={target_s}s seed={seed} lyrics={lyr_mode}{mode}")

        # final gate right before taking the card
        if hold_reason():
            continue
        # max_duration is a cap the encoder undershoots — never let it bind;
        # length is driven by the caption's stated duration and arc instead
        path, status = generate(caption, lyrics, seed, 300, steps_now)
        if not path:
            log(f"generation {status}; backoff {BACKOFF}s")
            time.sleep(BACKOFF)
            continue

        dur = duration_of(path)
        # Absolute stub floor, NOT target-coupled: the model's natural takes
        # run ~60-135s however rich the caption; a floor pinned to the
        # aspirational target rejects the whole distribution and starves the
        # radio. Targets still pull length upward via the caption text.
        if dur < MIN_TAKE_S:
            os.unlink(path)
            cool_vein(vein)
            record_verdict(vein, False, dur=dur)
            log(f"REJECT-stub {card['name']} {dur:.0f}s "
                f"(< {MIN_TAKE_S}s) — vein cools {COOLDOWN_S}s")
            if ONESHOT:
                return
            continue

        v = clap_embed(path)
        score = float(v @ critic[vein]["centroid"])
        thr_base = critic[vein]["threshold"]
        thr = thr_base - _relief.get(vein, 0.0)
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
                    "caption": caption, "lyrics": lyrics,
                    "created": int(time.time()), "consumed": False,
                }) + "\n")
            record_verdict(vein, True, dur=dur, score=score, thr=thr)
            eased = _relief.pop(vein, 0.0)
            log(f"ACCEPT {card['name']} {dur:.0f}s score {score:.3f} "
                f"(thr {thr:.3f}) -> {os.path.basename(dest)}"
                + (f" — bar restored to {thr_base:.3f}" if eased else ""))
        else:
            os.unlink(path)
            cool_vein(vein)
            record_verdict(vein, False, dur=dur, score=score, thr=thr)
            _relief[vein] = min(RELIEF_MAX,
                                _relief.get(vein, 0.0) + RELIEF_STEP)
            log(f"REJECT {card['name']} {dur:.0f}s score {score:.3f} "
                f"< thr {thr:.3f} — bar eases to "
                f"{thr_base - _relief[vein]:.3f}, vein cools {COOLDOWN_S}s")
        if ONESHOT:
            return


if __name__ == "__main__":
    main()
