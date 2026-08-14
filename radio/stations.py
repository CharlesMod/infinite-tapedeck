"""Stations: a station is a folder of songs treated as a mood — the whole
library, or a dozen hand-picked tracks — with its own analysis namespace,
tank, feedback, and keepers. The radio always plays exactly one station.

The original corpus keeps its legacy paths (the captioner may be writing
there at any moment); it is registered as the station "full-library" whose
paths map onto the old layout. New stations live under Music3/stations/<slug>/.
"""
import json
import os
import re
import time

BASE = os.environ.get("TAPEDECK_BASE") or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
REGISTRY = f"{BASE}/radio/stations.json"

LEGACY_SLUG = "full-library"
LEGACY = {
    "source": f"{BASE}/library",
    "analysis": f"{BASE}/analysis",
    "tank": f"{BASE}/radio/tank",
    "meta": f"{BASE}/radio/tank_meta.jsonl",
    "weights": f"{BASE}/radio/weights.json",
    "keepers": f"{BASE}/radio/keepers",
}


def _load():
    if os.path.exists(REGISTRY):
        try:
            with open(REGISTRY) as f:
                return json.load(f)
        except Exception:
            pass
    return {"active": LEGACY_SLUG,
            "stations": {LEGACY_SLUG: {"name": "Full Library",
                                       "source": LEGACY["source"],
                                       "created": int(time.time())}}}


def _save(reg):
    tmp = REGISTRY + ".tmp"
    with open(tmp, "w") as f:
        json.dump(reg, f, indent=1)
    os.replace(tmp, REGISTRY)


def slugify(name):
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return s or f"station-{int(time.time())}"


def paths(slug):
    """Every path a component needs, resolved for one station."""
    if slug == LEGACY_SLUG:
        return dict(LEGACY)
    root = f"{BASE}/stations/{slug}"
    reg = _load()
    src = reg["stations"].get(slug, {}).get("source", "")
    return {
        "source": src,
        "analysis": f"{root}/analysis",
        "tank": f"{root}/tank",
        "meta": f"{root}/tank_meta.jsonl",
        "weights": f"{root}/weights.json",
        "keepers": f"{root}/keepers",
    }


def listing():
    reg = _load()
    out = []
    for slug, meta in reg["stations"].items():
        p = paths(slug)
        cards = os.path.join(p["analysis"], "essence_cards.json")
        out.append({
            "slug": slug, "name": meta.get("name", slug),
            "source": meta.get("source", p["source"]),
            "ready": os.path.exists(cards),
            "active": slug == reg.get("active"),
        })
    return out


def active():
    return _load().get("active", LEGACY_SLUG)


def set_active(slug):
    reg = _load()
    if slug not in reg["stations"]:
        raise KeyError(slug)
    reg["active"] = slug
    _save(reg)


def create(name, source):
    """Register a new station and make its directories. Returns the slug."""
    source = os.path.realpath(os.path.expanduser(source))
    if not os.path.isdir(source):
        raise NotADirectoryError(source)
    reg = _load()
    slug = slugify(name)
    if slug in reg["stations"]:
        base = slug
        n = 2
        while f"{base}-{n}" in reg["stations"]:
            n += 1
        slug = f"{base}-{n}"
    reg["stations"][slug] = {"name": name, "source": source,
                             "created": int(time.time())}
    _save(reg)
    p = paths(slug)
    for d in (p["analysis"], p["tank"], p["keepers"]):
        os.makedirs(d, exist_ok=True)
    return slug


def tank_level(slug):
    """Unconsumed audio-seconds banked for a station — the radio's fuel gauge.
    Used by the capture pipeline to time-slice the GPU against generation."""
    p = paths(slug)
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
                elif not m.get("event") and "id" in m:
                    recs[m["id"]] = m
    total = 0.0
    for rid, m in recs.items():
        if rid not in consumed and not m.get("consumed"):
            total += m.get("duration_s", 0.0)
    return total


def ensure_registry():
    if not os.path.exists(REGISTRY):
        _save(_load())
