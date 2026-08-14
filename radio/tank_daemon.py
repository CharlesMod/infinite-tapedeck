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
        "steps": 30, "tank_target_s": 10800, "caption_batch": 5}
try:
    with open(f"{BASE}/radio/config.json") as _f:
        _cfg.update(json.load(_f))
except FileNotFoundError:
    pass
M3 = _cfg["comfy_host"]
SIBLINGS = list(_cfg["sibling_hosts"])
# Single-machine default: the LLM (llama-swap et al.) shares the one card
# with the generator. Captions and lyrics are written in batches per LLM
# residency so the swap tax amortizes. llm_base=None disables the LLM and
# the sampler falls back to essence-card seeds.
LLM_URL = (_cfg["llm_base"] or "").rstrip("/") or None
LLM_MODEL = _cfg["llm_model"]
CAPTION_BATCH = int(_cfg["caption_batch"])
MIN_TAKE_S = int(_cfg.get("min_take_s", 45))

STEPS = int(_cfg["steps"])
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


def log(msg):
    print(f"[tank] {msg}", flush=True)


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
    """llm_model unset -> first model the endpoint offers."""
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


def refill_bundles(cards, per, weights_path):
    """One LLM residency, several bundles: free the generator, write
    CAPTION_BATCH caption/lyric sets across the neediest veins, unload."""
    sim = dict(per)
    picks = []
    for _ in range(CAPTION_BATCH):
        v = pick_vein(cards, sim, weights_path)
        sim[v] = sim.get(v, 0.0) + 200.0  # assume a take lands, move on
        picks.append(v)
    free_music3()
    out = []
    for v in picks:
        card = cards[v]
        caption, bpm, target_s = sample_caption(v, card)
        lyr = (sample_lyrics(card)
               if "lyrics" in card.get("vocals", "").lower() else "")
        out.append((v, caption, lyr, bpm, target_s))
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

def build_graph(caption, lyrics, seed, max_duration):
    return {
        "1": {"class_type": "CLIPLoader", "inputs": {
            "clip_name": "minimax_music3_text_encoder_pruned_int8_convrot.safetensors",
            "type": "minimax", "device": "default"}},
        "2": {"class_type": "UNETLoader", "inputs": {
            "unet_name": "minimax_music3_dit_fp16.safetensors",
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
            "latent_image": ["6", 0], "seed": seed, "steps": STEPS,
            "cfg": CFG, "sampler_name": "euler", "scheduler": "simple",
            "denoise": 1.0}},
        "8": {"class_type": "VAEDecodeAudioTiled", "inputs": {
            "samples": ["7", 0], "vae": ["3", 0],
            "tile_size": 1536, "overlap": 64}},
        "9": {"class_type": "SaveAudioMP3", "inputs": {
            "audio": ["8", 0], "filename_prefix": "radio_tank/take",
            "quality": "V0"}},
    }


def generate(caption, lyrics, seed, max_duration):
    pid = http_json(M3 + "/prompt",
                    {"prompt": build_graph(caption, lyrics, seed,
                                           max_duration)})["prompt_id"]
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
    total = 0.0
    for rid, m in recs.items():
        if rid in consumed or m.get("consumed"):
            continue
        per[m["vein"]] = per.get(m["vein"], 0.0) + m["duration_s"]
        total += m["duration_s"]
    return total, per


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


def pick_vein(cards, per, weights_path):
    """Most-underfilled vein by RELATIVE deficit, so early fills interleave
    across veins instead of grinding the largest one to target first.
    Recently-rejected veins sit out a cooldown — a vein that keeps missing
    must not monopolize the card while the others could be banking takes."""
    w = effective_weights(cards, weights_path)
    now = time.time()
    deficits = {}
    for label in cards:
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
    bundles = []

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
                f"target {TANK_TARGET_S/3600:.1f}h, steps {STEPS}")
            last_slug = slug
            bundles = []  # bundles are per-station; stale ones are wrong moods

        janitor(p["meta"], p["tank"], p["keepers"])
        total, per = tank_seconds(p["meta"])
        if total >= TANK_TARGET_S:
            log(f"tank full ({total/3600:.2f}h)")
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
            log(f"writing {CAPTION_BATCH} bundles (LLM residency)")
            bundles = refill_bundles(cards, per, p["weights"])
        vein, caption, lyrics, bpm, target_s = bundles.pop(0)
        card = cards[vein]
        if TEST_SHORT:
            target_s = 45
        seed = random.randrange(2 ** 31)
        log(f"generating: vein={card['name']} bpm={bpm} "
            f"len<={target_s}s seed={seed} lyrics={'yes' if lyrics else 'no'}")

        # final gate right before taking the card
        if hold_reason():
            continue
        # max_duration is a cap the encoder undershoots — never let it bind;
        # length is driven by the caption's stated duration and arc instead
        path, status = generate(caption, lyrics, seed, 300)
        if not path:
            log(f"generation {status}; backoff {BACKOFF}s")
            time.sleep(BACKOFF)
            continue

        dur = duration_of(path)
        # Absolute stub floor, NOT target-coupled: natural takes run shorter
        # than aspirational targets however rich the caption; a target-coupled
        # floor rejects the whole distribution and starves the radio.
        if dur < MIN_TAKE_S:
            os.unlink(path)
            cool_vein(vein)
            log(f"REJECT-stub {card['name']} {dur:.0f}s "
                f"(< {MIN_TAKE_S}s) — vein cools {COOLDOWN_S}s")
            if ONESHOT:
                return
            continue

        v = clap_embed(path)
        score = float(v @ critic[vein]["centroid"])
        thr = critic[vein]["threshold"]
        if score >= thr:
            track_id = f"{int(time.time())}_{seed}"
            dest = os.path.join(p["tank"], f"v{vein}__{track_id}.mp3")
            os.replace(path, dest)
            with open(p["meta"], "a") as f:
                f.write(json.dumps({
                    "id": track_id, "vein": vein, "vein_name": card["name"],
                    "file": os.path.basename(dest), "duration_s": dur,
                    "score": round(score, 3), "threshold": round(thr, 3),
                    "bpm": bpm, "seed": seed, "steps": STEPS,
                    "caption": caption, "lyrics": lyrics,
                    "created": int(time.time()), "consumed": False,
                }) + "\n")
            log(f"ACCEPT {card['name']} {dur:.0f}s score {score:.3f} "
                f"(thr {thr:.3f}) -> {os.path.basename(dest)}")
        else:
            os.unlink(path)
            cool_vein(vein)
            log(f"REJECT {card['name']} {dur:.0f}s score {score:.3f} "
                f"< thr {thr:.3f} — vein cools {COOLDOWN_S}s")
        if ONESHOT:
            return


if __name__ == "__main__":
    main()
