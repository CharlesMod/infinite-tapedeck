#!/usr/bin/env python3
"""Inventory the mirrored library: per-file format facts via PyAV, aggregate
census, and a sanity flag for anything undecodable — better to find those now
than mid-analysis.

Usage: venv/bin/python analysis/inventory.py [library_dir] [out_json]
"""
import json
import os
import sys

import av

import os as _os
BASE = _os.environ.get("TAPEDECK_BASE") or _os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))


LIBRARY = sys.argv[1] if len(sys.argv) > 1 else f"{BASE}/library"
OUT = sys.argv[2] if len(sys.argv) > 2 else f"{BASE}/analysis/inventory.json"
AUDIO_EXTS = (".flac", ".mp3", ".m4a", ".ogg", ".opus", ".wav", ".wma", ".aac", ".aiff")


def probe(path):
    with av.open(path) as c:
        s = c.streams.audio[0]
        return {
            "duration_s": round(float(c.duration) / av.time_base, 1) if c.duration else None,
            "codec": s.codec_context.name,
            "sample_rate": s.rate,
            "channels": s.channels,
            "bit_rate": c.bit_rate or None,
        }


def main():
    tracks, failures = {}, []
    lrc, other = 0, 0
    for root, _, files in os.walk(LIBRARY):
        for name in sorted(files):
            path = os.path.join(root, name)
            rel = os.path.relpath(path, LIBRARY)
            low = name.lower()
            if low.endswith(AUDIO_EXTS):
                try:
                    tracks[rel] = {"bytes": os.path.getsize(path), **probe(path)}
                except Exception as e:
                    failures.append({"path": rel, "error": repr(e)[:120]})
            elif low.endswith(".lrc"):
                lrc += 1
            else:
                other += 1

    if not tracks:
        # First stage of the capture: catching an empty or wrong folder here
        # saves the user three stages of increasingly cryptic failures.
        print(f"NO AUDIO FOUND under {LIBRARY}", flush=True)
        if failures:
            print(f"({len(failures)} file(s) matched an audio extension but "
                  f"would not decode — first: {failures[0]['path']}: "
                  f"{failures[0]['error']})", flush=True)
        print(f"Put music there (or point the station at another folder) and "
              f"re-run.\nRecognized extensions: {', '.join(AUDIO_EXTS)}",
              flush=True)
        sys.exit(1)

    durs = [t["duration_s"] for t in tracks.values() if t["duration_s"]]
    hours = sum(durs) / 3600
    # duration histogram in minutes
    hist = {}
    for d in durs:
        b = f"{int(d // 60)}-{int(d // 60) + 1}min"
        hist[b] = hist.get(b, 0) + 1

    summary = {
        "tracks": len(tracks),
        "undecodable": len(failures),
        "lrc_files": lrc,
        "other_files": other,
        "total_hours": round(hours, 1),
        "mean_duration_s": round(sum(durs) / len(durs), 1) if durs else 0,
        "codecs": {},
        "duration_hist": dict(sorted(hist.items(),
                                     key=lambda kv: int(kv[0].split("-")[0]))),
    }
    for t in tracks.values():
        summary["codecs"][t["codec"]] = summary["codecs"].get(t["codec"], 0) + 1

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump({"summary": summary, "tracks": tracks, "failures": failures},
                  f, indent=1)
    print(json.dumps(summary, indent=1))
    for fl in failures:
        print("UNDECODABLE:", fl["path"], fl["error"])


if __name__ == "__main__":
    main()
