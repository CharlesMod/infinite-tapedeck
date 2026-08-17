#!/usr/bin/env python3
"""Cluster the corpus into taste veins from CLAP embeddings, then attach
deterministic-feature stats per vein. Density-based first (HDBSCAN) so the
data decides how many veins exist; k-means silhouette sweep as fallback if
density finds nothing. Noise points stay labeled — one-off tracks are
information, not failures — but get a nearest-vein pointer.

Output: analysis/veins.json + analysis/veins_report.md (skeleton for the
essence cards; prose distillation happens on top of this).

Usage: venv/bin/python analysis/cluster.py
"""
import json
import os
import sys
from collections import Counter

import numpy as np

import os as _os
BASE = _os.environ.get("TAPEDECK_BASE") or _os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))


LIB = sys.argv[1] if len(sys.argv) > 1 else f"{BASE}/library"
A = sys.argv[2] if len(sys.argv) > 2 else f"{BASE}/analysis"

# Below this corpus size, clustering is noise: the folder IS the mood.
SINGLE_VEIN_BELOW = 24

# The three ways a track can carry its energy across its length. Kept in one
# place because the sampler in radio/tank_daemon.py has to agree with it.
ARC_SHAPES = ("mid_peak", "builds_to_end", "front_loaded")


def arc_shape(thirds):
    """Which shape a single track's first/middle/last-third energy describes."""
    a, b, c = thirds
    if b >= a and b >= c:
        return "mid_peak"
    if c >= b >= a:
        return "builds_to_end"
    return "front_loaded"


def arc_distribution(thirds_rows):
    """What share of a vein's tracks does each shape. Every shape is present
    as a key even at 0.0, so a consumer can weight over it without guessing
    which names exist."""
    counts = Counter(arc_shape(row) for row in thirds_rows)
    n = sum(counts.values()) or 1
    return {s: round(counts.get(s, 0) / n, 3) for s in ARC_SHAPES}


def load():
    try:
        emb = np.load(os.path.join(A, "embeddings.npy"))
        with open(os.path.join(A, "embeddings_keys.json")) as f:
            keys = json.load(f)
    except FileNotFoundError as e:
        sys.exit(f"missing {os.path.basename(e.filename)} — the embeddings "
                 f"stage did not finish; re-run the capture.")
    feats = {}
    try:
        with open(os.path.join(A, "features.jsonl")) as f:
            for line in f:
                rec = json.loads(line)
                feats[rec["path"]] = rec
    except FileNotFoundError:
        sys.exit("missing features.jsonl — the features stage did not "
                 "finish; re-run the capture.")
    return emb, keys, feats


def has_lyrics(rel):
    stem = os.path.splitext(os.path.join(LIB, rel))[0]
    return os.path.exists(stem + ".lrc")


def cluster(emb):
    if emb.shape[0] < SINGLE_VEIN_BELOW:
        return (np.zeros(emb.shape[0], dtype=int),
                f"single-vein ({emb.shape[0]} tracks — the folder is the mood)")

    try:
        from sklearn.cluster import HDBSCAN, KMeans
        from sklearn.metrics import silhouette_score
    except ImportError:
        sys.exit(f"scikit-learn is required to cluster {emb.shape[0]} tracks "
                 "into veins (libraries under "
                 f"{SINGLE_VEIN_BELOW} tracks skip clustering and do not "
                 "need it).\n  pip install scikit-learn")

    h = HDBSCAN(min_cluster_size=10, min_samples=5, metric="euclidean")
    labels = h.fit_predict(emb)
    n = len(set(labels)) - (1 if -1 in labels else 0)
    noise = int((labels == -1).sum())
    # density failure: everything is one blob or everything is noise
    if n >= 3 and noise < len(labels) * 0.5:
        return labels, f"hdbscan(min_cluster_size=10): {n} veins, {noise} noise"

    best = (-1.0, None, None)
    for k in range(4, 13):
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(emb)
        s = silhouette_score(emb, km.labels_)
        if s > best[0]:
            best = (s, km.labels_, k)
    return best[1], f"kmeans(k={best[2]}, silhouette={best[0]:.3f}) [hdbscan fallback]"


def vein_stats(idx, keys, feats, emb, labels, label):
    members = [i for i in idx if labels[i] == label]
    vecs = emb[members]
    centroid = vecs.mean(axis=0)
    centroid /= np.linalg.norm(centroid) + 1e-9
    sims = vecs @ centroid
    order = np.argsort(-sims)
    ranked = [keys[members[i]] for i in order]

    fs = [feats[k] for k in ranked if k in feats]
    bpm = [f["tempo_bpm"] for f in fs]
    cent = [f["centroid_hz"] for f in fs]
    on = [f["onset_per_s"] for f in fs]
    dur = [f["duration_s"] for f in fs]
    keys_c = Counter(f["key"] for f in fs).most_common(4)
    thirds = np.array([f["energy_thirds"] for f in fs])
    # only a real directory component names an artist. A flat folder of
    # loose files would otherwise report filenames as "artists", and those
    # get quoted verbatim into generation prompts.
    artists = Counter(k.split(os.sep)[0] for k in ranked
                      if os.sep in k).most_common(6)
    lrc = sum(1 for k in ranked if has_lyrics(k))

    return {
        "size": len(members),
        "vocal_share": round(lrc / len(ranked), 2) if ranked else 0,
        "bpm": {"median": float(np.median(bpm)), "iqr": [float(np.percentile(bpm, 25)),
                                                         float(np.percentile(bpm, 75))]},
        "duration_median_s": float(np.median(dur)),
        "centroid_hz_median": float(np.median(cent)),
        "onset_per_s_median": float(np.median(on)),
        # The mean arc is kept because other stages read it, but it is a poor
        # description of a vein: averaging 80+ tracks cancels the per-song
        # variation and leaves only the universal "intros are quieter", so
        # every vein converges on mid-peak (measured: all four veins of a
        # 443-track corpus, a label that then differentiates nothing).
        # energy_shapes is the honest version — what share of the vein's own
        # tracks actually does each thing — so generation can span the same
        # range instead of always being told the mode.
        "energy_arc": [round(float(x), 4) for x in thirds.mean(axis=0)],
        "energy_shapes": arc_distribution(thirds),
        "top_keys": keys_c,
        "top_artists": artists,
        "central_tracks": ranked[:10],
        "all_tracks": ranked,
        "centroid": [round(float(x), 5) for x in centroid],
    }


def main():
    emb, keys, feats = load()
    assert emb.shape[0] == len(keys), "embeddings/keys misaligned"
    # tracks voted off the station: analyzed, but not part of its essence
    ex_path = os.path.join(A, "excluded.json")
    excluded = set()
    if os.path.exists(ex_path):
        with open(ex_path) as f:
            excluded = set(json.load(f))
    # a track needs BOTH an embedding and a feature row to describe a vein;
    # one stage failing on a file must not poison the stats with NaNs
    mask = [i for i, k in enumerate(keys)
            if k not in excluded and k in feats]
    if len(mask) != len(keys):
        dropped = len(keys) - len(mask)
        no_feats = sum(1 for k in keys if k not in feats and k not in excluded)
        print(f"using {len(mask)}/{len(keys)} tracks "
              f"({dropped} dropped: {len(excluded & set(keys))} excluded, "
              f"{no_feats} missing features)")
        emb = emb[mask]
        keys = [keys[i] for i in mask]
    if not keys:
        sys.exit("no tracks left to cluster — every track was excluded or "
                 "missing features. Re-run the capture.")
    labels, method = cluster(emb)
    print("method:", method)

    idx = list(range(len(keys)))
    out = {"method": method, "veins": {}, "noise": []}
    for label in sorted(set(labels)):
        if label == -1:
            continue
        out["veins"][str(label)] = vein_stats(idx, keys, feats, emb, labels, label)

    # noise: nearest vein by centroid for context
    cents = {l: np.array(v["centroid"]) for l, v in out["veins"].items()}
    for i in idx:
        if labels[i] == -1:
            sims = {l: float(emb[i] @ c) for l, c in cents.items()}
            near = max(sims, key=sims.get)
            out["noise"].append({"path": keys[i], "nearest_vein": near,
                                 "sim": round(sims[near], 3)})

    with open(os.path.join(A, "veins.json"), "w") as f:
        json.dump(out, f, indent=1)

    lines = [f"# Taste veins — {method}", ""]
    for l, v in sorted(out["veins"].items(), key=lambda kv: -kv[1]["size"]):
        lines += [f"## Vein {l} — {v['size']} tracks "
                  f"({int(v['vocal_share']*100)}% with lyrics)",
                  f"- BPM median {v['bpm']['median']:.0f} "
                  f"(IQR {v['bpm']['iqr'][0]:.0f}–{v['bpm']['iqr'][1]:.0f}) | "
                  f"onsets/s {v['onset_per_s_median']:.1f} | "
                  f"spectral centroid {v['centroid_hz_median']:.0f} Hz | "
                  f"median length {v['duration_median_s']/60:.1f} min",
                  f"- energy arc (thirds): {v['energy_arc']}",
                  f"- keys: {', '.join(f'{k} ×{c}' for k, c in v['top_keys'])}",
                  f"- artists: {', '.join(f'{a} ×{c}' for a, c in v['top_artists'])}",
                  "- most central:"]
        lines += [f"  - {t}" for t in v["central_tracks"]]
        lines += [""]
    if out["noise"]:
        lines += [f"## Unclustered ({len(out['noise'])} one-offs)", ""]
        lines += [f"- {n['path']} → nearest vein {n['nearest_vein']} ({n['sim']})"
                  for n in out["noise"][:20]]
    with open(os.path.join(A, "veins_report.md"), "w") as f:
        f.write("\n".join(lines))
    print(f"{len(out['veins'])} veins, {len(out['noise'])} unclustered "
          f"-> veins.json, veins_report.md")
    n = len(out["veins"])
    print(f"RESULT {n} vein{'' if n == 1 else 's'} · {method}"
          + (f" · {len(out['noise'])} one-offs" if out["noise"] else ""),
          flush=True)


if __name__ == "__main__":
    main()
