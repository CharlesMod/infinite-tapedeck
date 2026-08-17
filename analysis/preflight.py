#!/usr/bin/env python3
"""Check an install before it is asked to do any real work.

Every check answers a question the capture pipeline or the radio would
otherwise answer with a stack trace an hour in: is this the interpreter that
has the libraries, can it decode audio, is ComfyUI up, are the model files
where the graph expects them, is there music to capture.

Usage: python analysis/preflight.py [--station SLUG]
Exit code 0 = ready to capture, 1 = something is missing.
"""
import json
import os
import shutil
import subprocess
import sys

BASE = os.environ.get("TAPEDECK_BASE") or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))

OK, WARN, BAD = "  ok  ", " warn ", " FAIL "
_fails = 0
_warns = 0


def report(level, what, detail=""):
    global _fails, _warns
    if level is BAD:
        _fails += 1
    elif level is WARN:
        _warns += 1
    print(f"[{level}] {what}" + (f"\n         {detail}" if detail else ""))


def check_python():
    v = sys.version_info
    detail = f"{sys.executable}"
    if v < (3, 10):
        report(BAD, f"python {v.major}.{v.minor} — needs 3.10+", detail)
    else:
        report(OK, f"python {v.major}.{v.minor}", detail)


def check_imports():
    """The deps, in the interpreter that will actually run the stages.

    This is the check that matters most: the usual install failure is a venv
    that has the libraries and a `python` on PATH that does not, or the
    reverse. Run this with the same command you will use to capture."""
    required = [
        ("numpy", "numpy", "features, embeddings, clustering"),
        ("librosa", "librosa", "audio analysis"),
        ("soundfile", "soundfile", "audio I/O"),
        ("av", "av", "container probing"),
        ("torch", "torch", "CLAP embeddings and the critic"),
        ("transformers", "transformers", "CLAP model"),
        ("sklearn", "scikit-learn", "clustering libraries of 24+ tracks"),
    ]
    optional = [
        ("bitsandbytes", "bitsandbytes", "AI captions (--with-captions)"),
        ("accelerate", "accelerate", "AI captions (--with-captions)"),
    ]
    import importlib.util
    for mod, pkg, why in required:
        if importlib.util.find_spec(mod) is None:
            report(BAD, f"{pkg} missing — {why}", f"pip install {pkg}")
        else:
            report(OK, pkg)
    for mod, pkg, why in optional:
        if importlib.util.find_spec(mod) is None:
            report(WARN, f"{pkg} missing — needed only for {why}",
                   f"pip install {pkg}")
        else:
            report(OK, f"{pkg} (optional)")


def check_decoders(exts):
    """Can this machine actually decode the formats in the library?

    librosa reads through libsndfile first and falls back to ffmpeg. Modern
    libsndfile covers flac/mp3/ogg/wav/aiff on its own, so ffmpeg only
    matters for the container formats it does not implement — worth checking
    against the real library rather than warning everybody about ffmpeg."""
    try:
        import soundfile
        native = {f".{f.lower()}" for f in soundfile.available_formats()}
    except ImportError:
        return
    native |= {".ogg"}  # libsndfile reports OGG for vorbis
    needs_ffmpeg = sorted(e for e in exts if e not in native)
    if not needs_ffmpeg:
        if exts:
            report(OK, f"audio decoding ({', '.join(sorted(exts))} handled "
                       "by libsndfile)")
        return
    if shutil.which("ffmpeg"):
        report(OK, f"ffmpeg on PATH (needed for {', '.join(needs_ffmpeg)})")
    else:
        report(BAD, f"no decoder for {', '.join(needs_ffmpeg)} in your library",
               "libsndfile cannot read these and ffmpeg is not on PATH — "
               "every one of those tracks will fail.\n         "
               "Install ffmpeg, or convert them to flac/mp3/ogg.")


def check_torch_gpu():
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
        level = OK if vram >= 15 else WARN
        report(level, f"GPU: {name} ({vram:.0f} GB VRAM)",
               "" if vram >= 15 else "16 GB is the tested minimum")
    else:
        report(WARN, "torch cannot see a GPU",
               "analysis will run on CPU (slow); generation needs the GPU")


def _comfy_host():
    try:
        with open(f"{BASE}/radio/config.json") as f:
            return json.load(f).get("comfy_host") or "http://127.0.0.1:8188"
    except (OSError, ValueError):
        return "http://127.0.0.1:8188"


def check_comfy():
    """Only the radio needs ComfyUI; a capture does not. Warn, never fail."""
    import urllib.request
    host = _comfy_host()
    try:
        with urllib.request.urlopen(host + "/system_stats", timeout=5) as r:
            json.load(r)
        report(OK, f"ComfyUI reachable at {host}")
    except Exception as e:
        report(WARN, f"ComfyUI not reachable at {host}",
               f"{e!r:.60}\n         Needed to generate, not to capture. "
               "Set comfy_host in radio/config.json if it runs elsewhere.")


def check_models():
    """The three files build_graph() names. A missing one surfaces as a
    ComfyUI validation error per generation, forever."""
    comfy = os.environ.get("COMFYUI_DIR") or f"{BASE}/ComfyUI"
    if not os.path.isdir(comfy):
        report(WARN, f"no ComfyUI directory at {comfy}",
               "Set COMFYUI_DIR to check the Music 3 model files.")
        return
    wanted = [
        ("models/diffusion_models", "minimax_music3_dit_fp16.safetensors"),
        ("models/text_encoders",
         "minimax_music3_text_encoder_pruned_int8_convrot.safetensors"),
        ("models/vae", "minimax_music3_dav.safetensors"),
    ]
    for sub, name in wanted:
        d = os.path.join(comfy, sub)
        if os.path.exists(os.path.join(d, name)):
            report(OK, f"{name}")
        elif os.path.isdir(d) and any(f.startswith("minimax_music3")
                                      for f in os.listdir(d)):
            found = [f for f in os.listdir(d) if f.startswith("minimax_music3")]
            report(WARN, f"{name} not found in {sub}",
                   f"but found {', '.join(found)} — set dit_model in "
                   "radio/config.json if you use a different quantization")
        else:
            report(BAD, f"{name} missing from {sub}",
                   "see docs/INSTALL.md step 3")


def check_library(slug):
    exts = (".flac", ".mp3", ".m4a", ".ogg", ".opus", ".wav", ".wma",
            ".aac", ".aiff")
    sys.path.insert(0, f"{BASE}/radio")
    try:
        import stations
        src = stations.paths(slug or stations.active())["source"]
    except Exception:
        src = f"{BASE}/library"
    if not os.path.isdir(src):
        report(BAD, f"library folder does not exist: {src}",
               "create it and put music in it, or create a station "
               "pointing somewhere else")
        return set()
    n = 0
    found = set()
    for _, _, files in os.walk(src):
        for f in files:
            e = os.path.splitext(f.lower())[1]
            if e in exts:
                n += 1
                found.add(e)
    if n == 0:
        report(BAD, f"no audio files under {src}",
               f"recognized extensions: {', '.join(exts)}")
    elif n < 8:
        report(WARN, f"{n} track(s) in {src}",
               "a station this small gives the critic very little to "
               "calibrate against; 12+ works better")
    else:
        report(OK, f"{n} tracks in {src}")
    return found


def check_disk():
    free = shutil.disk_usage(BASE).free / 2 ** 30
    if free < 5:
        report(BAD, f"{free:.1f} GB free on the tapedeck volume",
               "generation stops below 2 GB")
    else:
        report(OK, f"{free:.0f} GB free")


def main():
    slug = (sys.argv[sys.argv.index("--station") + 1]
            if "--station" in sys.argv else None)
    print(f"tapedeck preflight — base: {BASE}\n")
    check_python()
    check_imports()
    check_torch_gpu()
    check_disk()
    exts = check_library(slug) or set()
    check_decoders(exts)
    check_comfy()
    check_models()

    print()
    if _fails:
        print(f"{_fails} blocking problem(s)"
              + (f", {_warns} warning(s)" if _warns else "")
              + " — fix the FAIL lines above before capturing.")
        return 1
    print("ready to capture"
          + (f" ({_warns} warning(s) above)" if _warns else "") + ".")
    return 0


if __name__ == "__main__":
    sys.exit(main())
