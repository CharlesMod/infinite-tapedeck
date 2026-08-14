"""Music Studio: the radio deck for the infinite personal radio.

Pandora model: a *station* is a folder of songs treated as a mood — the whole
library or a dozen hand-picked tracks — with its own analysis, tank, feedback,
and keepers. The deck always plays the active station; the tank daemon fills
it. Creating a station from a folder kicks the capture pipeline; the rail in
the deck UI shows the setlist and capture progress at all times.

Persistence-first, deliberately unlike h3_studio: served tracks are marked
consumed (the tank refills behind them), keepers live forever, and the
feedback log is the whole point.
"""
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time

from aiohttp import web

from server import PromptServer

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
WEB_DIRECTORY = "./web"

def _find_base():
    env = os.environ.get("TAPEDECK_BASE")
    if env:
        return env
    marker = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "tapedeck_base.txt")
    if os.path.exists(marker):
        return open(marker).read().strip()
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BASE = _find_base()
sys.path.insert(0, f"{BASE}/radio")
import stations  # noqa: E402

ANALYSIS_SCRIPTS = f"{BASE}/analysis"
IMPORT_PROGRESS = f"{ANALYSIS_SCRIPTS}/import_progress.json"
VENV_PY = sys.executable
AUDIO_EXTS = (".flac", ".mp3", ".m4a", ".ogg", ".opus", ".wav", ".wma",
              ".aac", ".aiff")

FILE_RE = re.compile(r"^v\d+__\d+_\d+\.mp3$")
FEEDBACK_MULT = {"keep": 1.10, "dislike": 0.85, "skip": 0.97}

stations.ensure_registry()
_last_vein = None
_import_proc = None


def _p():
    """Paths for the active station."""
    return stations.paths(stations.active())


def _read_state(p):
    """Fold the station's event log into: track records, available tracks,
    and the consumed-id set (the played ledger)."""
    recs, consumed = {}, set()
    if os.path.exists(p["meta"]):
        with open(p["meta"]) as f:
            for line in f:
                try:
                    m = json.loads(line)
                except Exception:
                    continue
                if m.get("event") == "consumed":
                    consumed.add(m["id"])
                elif m.get("event"):
                    continue
                elif "id" in m:
                    recs[m["id"]] = m
    avail = [m for rid, m in recs.items()
             if rid not in consumed and not m.get("consumed")
             and os.path.exists(os.path.join(p["tank"], m["file"]))]
    return recs, avail, consumed


def _cards(p):
    with open(os.path.join(p["analysis"], "essence_cards.json")) as f:
        return json.load(f)["veins"]


def _weights(p):
    if os.path.exists(p["weights"]):
        try:
            with open(p["weights"]) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _append_event(p, obj):
    with open(p["meta"], "a") as f:
        f.write(json.dumps(obj) + "\n")


# ------------------------------------------------------------------ playback

@PromptServer.instance.routes.get("/music_studio/next")
async def next_track(request):
    global _last_vein
    p = _p()
    try:
        cards = _cards(p)
    except FileNotFoundError:
        return web.json_response({"empty": True,
                                  "hint": "station not captured yet"})
    mult = _weights(p)
    recs, avail, _ = _read_state(p)
    if not avail:
        # a listener is waiting on an empty spool: tell the daemon (it will
        # cut an emergency take), and NEVER sit silent — replay the archive
        # (keepers + consumed takes the janitor hasn't collected) meanwhile.
        try:
            with open(f"{BASE}/radio/LISTENER_WAITING", "w") as f:
                f.write(str(int(time.time())))
        except OSError:
            pass
        pool = []
        for t in recs.values():
            for root in (p["tank"], p["keepers"]):
                if os.path.exists(os.path.join(root, t["file"])):
                    pool.append(t)
                    break
        if pool:
            t = random.choice(pool)
            cards_r = {}
            try:
                cards_r = _cards(p)
            except FileNotFoundError:
                pass
            return web.json_response({
                "id": t["id"], "vein": t["vein"], "replay": True,
                "vein_name": t.get("vein_name",
                                   cards_r.get(t["vein"], {}).get("name", "?")),
                "url": f"/music_studio/audio/{t['file']}",
                "duration_s": t["duration_s"], "bpm": t.get("bpm"),
                "score": t.get("score"), "caption": t.get("caption", ""),
                "lyrics": t.get("lyrics", ""), "left_in_tank": 0,
            })
        return web.json_response({"empty": True,
                                  "hint": "fresh tape is being synthesized "
                                          "— give the deck a few minutes"})

    by_vein = {}
    for m in avail:
        by_vein.setdefault(m["vein"], []).append(m)

    pool = {v: cards.get(v, {"weight": 0.1})["weight"] * float(mult.get(v, 1.0))
            for v in by_vein}
    if _last_vein in pool and len(pool) > 1:
        pool[_last_vein] *= 0.25
    total = sum(pool.values())
    r = random.uniform(0, total)
    acc = 0.0
    vein = next(iter(pool))
    for v, w in pool.items():
        acc += w
        if r <= acc:
            vein = v
            break
    _last_vein = vein

    track = random.choice(by_vein[vein])
    _append_event(p, {"event": "consumed", "id": track["id"],
                      "ts": int(time.time())})
    return web.json_response({
        "id": track["id"], "vein": vein,
        "vein_name": track.get("vein_name",
                               cards.get(vein, {}).get("name", f"Vein {vein}")),
        "url": f"/music_studio/audio/{track['file']}",
        "duration_s": track["duration_s"], "bpm": track.get("bpm"),
        "score": track.get("score"),
        "caption": track.get("caption", ""),
        "lyrics": track.get("lyrics", ""),
        "left_in_tank": len(avail) - 1,
    })


@PromptServer.instance.routes.get("/music_studio/audio/{name}")
async def audio(request):
    name = request.match_info["name"]
    if not FILE_RE.match(name):
        return web.json_response({"error": "rejected"}, status=400)
    p = _p()
    for root in (p["tank"], p["keepers"]):
        path = os.path.realpath(os.path.join(root, name))
        if os.path.dirname(path) == os.path.realpath(root) and os.path.exists(path):
            return web.FileResponse(path)
    return web.json_response({"error": "gone"}, status=404)


@PromptServer.instance.routes.post("/music_studio/feedback")
async def feedback(request):
    data = await request.json()
    action = data.get("action")
    tid = str(data.get("id") or "")
    if action not in FEEDBACK_MULT or not tid:
        return web.json_response({"error": "bad request"}, status=400)
    p = _p()
    recs, _, _ = _read_state(p)
    track = recs.get(tid)
    if not track:
        return web.json_response({"error": "unknown id"}, status=404)

    kept_as = None
    if action == "keep":
        os.makedirs(p["keepers"], exist_ok=True)
        src = os.path.join(p["tank"], track["file"])
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(p["keepers"], track["file"]))
            kept_as = track["file"]

    vein = track["vein"]
    mult = _weights(p)
    mult[vein] = round(min(3.0, max(0.3, float(mult.get(vein, 1.0))
                                    * FEEDBACK_MULT[action])), 4)
    with open(p["weights"], "w") as f:
        json.dump(mult, f, indent=1)
    _append_event(p, {"event": "feedback", "id": tid, "action": action,
                      "vein": vein, "ts": int(time.time())})
    return web.json_response({"ok": True, "vein_mult": mult[vein],
                              "kept_as": kept_as})


@PromptServer.instance.routes.get("/music_studio/history")
async def history(request):
    limit = min(int(request.query.get("limit", 20)), 100)
    p = _p()
    recs, order = {}, []
    if os.path.exists(p["meta"]):
        with open(p["meta"]) as f:
            for line in f:
                try:
                    m = json.loads(line)
                except Exception:
                    continue
                if m.get("event") == "consumed":
                    order.append(m["id"])
                elif not m.get("event") and "id" in m:
                    recs[m["id"]] = m
    try:
        cards = _cards(p)
    except FileNotFoundError:
        cards = {}
    out = []
    for tid in order[-limit:]:
        t = recs.get(tid)
        if not t:
            continue
        if not any(os.path.exists(os.path.join(p[k], t["file"]))
                   for k in ("tank", "keepers")):
            continue
        out.append({
            "id": t["id"], "vein": t["vein"],
            "vein_name": t.get("vein_name",
                               cards.get(t["vein"], {}).get("name", "?")),
            "url": f"/music_studio/audio/{t['file']}",
            "duration_s": t["duration_s"], "bpm": t.get("bpm"),
            "score": t.get("score"), "caption": t.get("caption", ""),
            "lyrics": t.get("lyrics", ""),
        })
    return web.json_response({"history": out})


@PromptServer.instance.routes.get("/music_studio/status")
async def status(request):
    p = _p()
    try:
        cards = _cards(p)
    except FileNotFoundError:
        cards = {}
    recs, avail, consumed = _read_state(p)
    per = {}
    for m in avail:
        v = per.setdefault(m["vein"], {"tracks": 0, "seconds": 0.0})
        v["tracks"] += 1
        v["seconds"] += m["duration_s"]
    played_n, played_s = 0, 0.0
    for rid in consumed:
        m = recs.get(rid)
        if m:
            played_n += 1
            played_s += m.get("duration_s", 0.0)
    keepers = len(os.listdir(p["keepers"])) if os.path.isdir(p["keepers"]) else 0
    return web.json_response({
        "station": stations.active(),
        "played": {"tracks": played_n, "minutes": round(played_s / 60, 1)},
        "veins": {v: {"name": c.get("name", f"Vein {v}"),
                      "tracks": per.get(v, {}).get("tracks", 0),
                      "minutes": round(per.get(v, {}).get("seconds", 0) / 60, 1),
                      "mult": _weights(p).get(v, 1.0)}
                  for v, c in cards.items()},
        "keepers": keepers,
    })


# ------------------------------------------------------------------ stations

def _card_owner():
    """Who is actually SPENDING the GPU right now — one card, several
    claimants, and the deck must show work, not liveness. Classified by
    per-process VRAM attribution."""
    mem = {"generation": 0, "dubbing": 0, "lyricist": 0, "other": 0}
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5).stdout
        for line in out.strip().splitlines():
            try:
                pid_s, mem_s = [x.strip() for x in line.split(",")]
                pid, mb = int(pid_s), int(mem_s)
                with open(f"/proc/{pid}/cmdline", "rb") as f:
                    cmd = f.read().decode(errors="replace")
            except (ValueError, OSError):
                continue
            if "caption_pass" in cmd:
                mem["dubbing"] += mb
            elif "llama-server" in cmd:
                mem["lyricist"] += mb
            elif pid == os.getpid():
                mem["generation"] += mb
            elif mb > 400:
                mem["other"] += mb
    except Exception:
        pass
    running, pending = PromptServer.instance.prompt_queue.get_current_queue()
    if running and mem["generation"] > 3000:
        return "generation", mem
    if mem["dubbing"] > 3000:
        return "dubbing", mem
    if mem["lyricist"] > 1500:
        return "lyricist", mem
    if mem["other"] > 3000:
        return "other", mem
    return "idle", mem


@PromptServer.instance.routes.get("/music_studio/deckstate")
async def deckstate(request):
    """The tank daemon's last words, their age, and the card's current
    claimant — the deck's answer to 'what is the machine doing right now
    and why am I waiting?'"""
    try:
        with open(f"{BASE}/radio/daemon_state.json") as f:
            st = json.load(f)
        st["age_s"] = max(0, int(time.time()) - int(st.get("ts", 0)))
    except Exception:
        st = {"msg": "daemon has not spoken yet", "age_s": None}
    owner, mem = _card_owner()
    st["owner"] = owner
    st["owner_mem"] = mem
    return web.json_response(st)


@PromptServer.instance.routes.get("/music_studio/gpu")
async def gpu(request):
    """Whole-card utilization and VRAM for the deck's meter bridge — any
    tenant counts (generation, captioner, games), which is the point:
    the user should always be able to see that the machine is working."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5).stdout.strip()
        util, used, total = [int(x) for x in out.split(",")]
        return web.json_response({"util": util, "vram_used_mb": used,
                                  "vram_total_mb": total})
    except Exception as e:
        return web.json_response({"error": repr(e)[:80]}, status=503)


@PromptServer.instance.routes.get("/music_studio/stations")
async def stations_list(request):
    return web.json_response({"stations": stations.listing(),
                              "active": stations.active()})


@PromptServer.instance.routes.post("/music_studio/station/select")
async def station_select(request):
    global _last_vein
    data = await request.json()
    try:
        stations.set_active(str(data.get("slug") or ""))
    except KeyError:
        return web.json_response({"error": "unknown station"}, status=404)
    _last_vein = None
    return web.json_response({"ok": True, "active": stations.active()})


@PromptServer.instance.routes.post("/music_studio/station/create")
async def station_create(request):
    """Register a folder as a new station and start capturing it."""
    data = await request.json()
    name = str(data.get("name") or "").strip()
    source = str(data.get("source") or "").strip()
    if not name or not source:
        return web.json_response({"error": "name and source required"},
                                 status=400)
    try:
        slug = stations.create(name, source)
    except NotADirectoryError as e:
        return web.json_response({"error": f"not a directory: {e}"}, status=400)
    err = _spawn_import(slug, bool(data.get("with_captions")))
    if err:
        return web.json_response({"ok": True, "slug": slug,
                                  "capture": f"not started: {err}"})
    return web.json_response({"ok": True, "slug": slug, "capture": "started"})


def _excluded_path(p):
    return os.path.join(p["analysis"], "excluded.json")


def _excluded(p):
    try:
        with open(_excluded_path(p)) as f:
            return set(json.load(f))
    except Exception:
        return set()


def _source_audio(src):
    out = []
    if os.path.isdir(src):
        for root, _, files in os.walk(src):
            for nm in sorted(files):
                if nm.lower().endswith(AUDIO_EXTS):
                    out.append(os.path.relpath(os.path.join(root, nm), src))
    return out


@PromptServer.instance.routes.get("/music_studio/station/scan")
async def station_scan(request):
    """Diff the station's folder against what has been analyzed — the
    Refresh button. New files show up after drops or manual copies."""
    p = _p()
    on_disk = set(_source_audio(p["source"]))
    analyzed = set()
    fpath = os.path.join(p["analysis"], "features.jsonl")
    if os.path.exists(fpath):
        with open(fpath) as f:
            for line in f:
                try:
                    analyzed.add(json.loads(line)["path"])
                except Exception:
                    continue
    return web.json_response({
        "total": len(on_disk),
        "analyzed": len(on_disk & analyzed),
        "new": sorted(on_disk - analyzed)[:200],
        "new_count": len(on_disk - analyzed),
        "removed_count": len(analyzed - on_disk),
        "excluded_count": len(_excluded(p)),
    })


@PromptServer.instance.routes.post("/music_studio/station/exclude")
async def station_exclude(request):
    """Vote a track off the station (or back on). Non-destructive: the file
    stays on disk; the track just stops counting toward the essence."""
    data = await request.json()
    rel = str(data.get("path") or "")
    restore = bool(data.get("restore"))
    p = _p()
    if not rel or os.path.isabs(rel) or ".." in rel.split(os.sep):
        return web.json_response({"error": "rejected"}, status=400)
    ex = _excluded(p)
    if restore:
        ex.discard(rel)
    else:
        ex.add(rel)
    os.makedirs(p["analysis"], exist_ok=True)
    with open(_excluded_path(p), "w") as f:
        json.dump(sorted(ex), f, indent=1)
    return web.json_response({"ok": True, "excluded_count": len(ex),
                              "recluster_hint": "run capture to re-derive the essence"})


@PromptServer.instance.routes.post("/music_studio/station/upload")
async def station_upload(request):
    """Drag-and-drop landing zone: multipart audio files written into the
    active station's source folder."""
    p = _p()
    src = p["source"]
    if not os.path.isdir(src):
        return web.json_response({"error": "station source missing"}, status=400)
    added, refused = [], []
    reader = await request.multipart()
    while True:
        part = await reader.next()
        if part is None:
            break
        name = os.path.basename(part.filename or "")
        if not name.lower().endswith(AUDIO_EXTS):
            refused.append(name or "unnamed")
            continue
        dest = os.path.join(src, name)
        stem, ext = os.path.splitext(dest)
        n = 2
        while os.path.exists(dest):
            dest = f"{stem}-{n}{ext}"
            n += 1
        size = 0
        with open(dest, "wb") as f:
            while True:
                chunk = await part.read_chunk(1 << 20)
                if not chunk:
                    break
                size += len(chunk)
                if size > 300 * (1 << 20):
                    f.close()
                    os.unlink(dest)
                    refused.append(name + " (too large)")
                    dest = None
                    break
                f.write(chunk)
        if dest:
            added.append(os.path.basename(dest))
    return web.json_response({"ok": True, "added": added, "refused": refused})


@PromptServer.instance.routes.post("/music_studio/import/pause")
async def import_pause(request):
    """Stop a running capture in a resumable way. Stages are incremental, so
    pausing costs at most the in-flight track — and unlike freezing the
    process, it actually releases the GPU."""
    st = _import_state()
    if st.get("state") != "running":
        return web.json_response({"ok": False, "state": st.get("state")})
    try:
        os.kill(int(st["pid"]), 15)
    except OSError as e:
        return web.json_response({"error": repr(e)[:80]}, status=500)
    for _ in range(40):  # wait for teardown so "paused" wins the state race
        if not _pid_alive(st["pid"]):
            break
        import asyncio
        await asyncio.sleep(0.5)
    st["state"] = "paused"
    with open(IMPORT_PROGRESS, "w") as f:
        json.dump(st, f)
    return web.json_response({"ok": True})


@PromptServer.instance.routes.post("/music_studio/import/resume")
async def import_resume(request):
    """Re-run the paused capture; resume keys make it continue, not restart."""
    st = _import_state()
    if st.get("state") not in ("paused", "cancelled", "failed", "died"):
        return web.json_response({"error": f"nothing to resume "
                                  f"(state: {st.get('state')})"}, status=409)
    slug = st.get("station")
    if not slug:
        return web.json_response({"error": "no station recorded"}, status=400)
    err = _spawn_import(slug, bool(st.get("with_captions")))
    if err:
        return web.json_response({"error": err}, status=409)
    return web.json_response({"ok": True, "station": slug})


@PromptServer.instance.routes.get("/music_studio/station/tracks")
async def station_tracks(request):
    """The active station's setlist: its source songs, with features when
    the capture has them."""
    p = _p()
    feats = {}
    fpath = os.path.join(p["analysis"], "features.jsonl")
    if os.path.exists(fpath):
        with open(fpath) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    feats[r["path"]] = r
                except Exception:
                    continue
    out = []
    src = p["source"]
    ex = _excluded(p)
    for rel in _source_audio(src):
        fx = feats.get(rel, {})
        out.append({"path": rel,
                    "bpm": fx.get("tempo_bpm"),
                    "key": fx.get("key"),
                    "duration_s": fx.get("duration_s"),
                    "analyzed": rel in feats,
                    "excluded": rel in ex})
    return web.json_response({"source": src, "tracks": out})


# -------------------------------------------------------------------- import

def _pid_alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


def _import_state():
    if not os.path.exists(IMPORT_PROGRESS):
        return {"state": "idle"}
    try:
        with open(IMPORT_PROGRESS) as f:
            st = json.load(f)
    except Exception:
        return {"state": "idle"}
    if st.get("state") == "running" and not _pid_alive(st.get("pid")):
        st["state"] = "died"
    return st


def _spawn_import(slug, with_captions):
    """Start the capture pipeline for a station. Returns error string or None."""
    global _import_proc
    if _import_state().get("state") == "running":
        return "an import is already running"
    cmd = [VENV_PY, f"{ANALYSIS_SCRIPTS}/import_pipeline.py", "--station", slug]
    if with_captions:
        cmd.append("--with-captions")
    # a capture must outlive this server: a plain child sits in our systemd
    # cgroup and dies with every service restart (start_new_session does NOT
    # escape a cgroup). A transient scope gets its own; plain child fallback
    # where systemd-run is unavailable.
    if shutil.which("systemd-run"):
        cmd = ["systemd-run", "--user", "--scope", "--collect", "-q"] + cmd
    log = open(f"{ANALYSIS_SCRIPTS}/import_run.log", "a")
    _import_proc = subprocess.Popen(cmd, stdout=log, stderr=log,
                                    start_new_session=True)
    return None


@PromptServer.instance.routes.post("/music_studio/import/start")
async def import_start(request):
    """Capture a directory. A path equal to an existing station's source
    re-captures that station; a new path becomes a new station named after
    its folder."""
    data = await request.json()
    source = os.path.realpath(os.path.expanduser(str(data.get("source") or "")))
    if not os.path.isdir(source):
        return web.json_response({"error": f"not a directory: {source}"},
                                 status=400)
    slug = None
    for st in stations.listing():
        if os.path.realpath(st["source"]) == source:
            slug = st["slug"]
            break
    if slug is None:
        slug = stations.create(os.path.basename(source) or "station", source)
    err = _spawn_import(slug, bool(data.get("with_captions")))
    if err:
        return web.json_response({"error": err}, status=409)
    return web.json_response({"ok": True, "station": slug})


@PromptServer.instance.routes.get("/music_studio/import/status")
async def import_status(request):
    return web.json_response(_import_state())


@PromptServer.instance.routes.post("/music_studio/import/cancel")
async def import_cancel(request):
    st = _import_state()
    if st.get("state") != "running":
        return web.json_response({"ok": False, "state": st.get("state")})
    try:
        os.kill(int(st["pid"]), 15)
        return web.json_response({"ok": True})
    except OSError as e:
        return web.json_response({"error": repr(e)[:80]}, status=500)


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
