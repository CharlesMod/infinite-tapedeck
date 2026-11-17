"""Consent for the AI listening pass (NVIDIA Music Flamingo).

The listening pass is ON by default — it is the single biggest quality
difference — but it downloads ~17 GB of NVIDIA's weights under NVIDIA's
licence, so it never runs until the user has said yes once:

  - the deck's "Accept & start listening pass" button, or
  - import_pipeline.py --accept-flamingo on the command line.

Until then a capture runs without captions and the deck shows the button.
The answer lives in radio/speed.json (the deck's settings file) under
"flamingo_consent". Every path that could start the download checks here:
the server's _spawn_import, import_pipeline.py and caption_pass.py itself.
"""
import json
import os

BASE = os.environ.get("TAPEDECK_BASE") or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
SETTINGS = f"{BASE}/radio/speed.json"
KEY = "flamingo_consent"
MODEL = "nvidia/music-flamingo-2601-hf"
MODEL_URL = f"https://huggingface.co/{MODEL}"
DOWNLOAD_GB = 17


def _read():
    try:
        with open(SETTINGS) as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def given():
    """True only after an explicit yes. Missing or false means no."""
    return _read().get(KEY) is True


def record(value=True):
    """Store the answer atomically, keeping the file's other settings."""
    d = _read()
    d[KEY] = bool(value)
    os.makedirs(os.path.dirname(SETTINGS), exist_ok=True)
    tmp = SETTINGS + ".tmp"
    with open(tmp, "w") as f:
        json.dump(d, f)
    os.replace(tmp, SETTINGS)
    return d[KEY]


def model_cached():
    cache = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    return os.path.isdir(os.path.join(
        cache, "hub", "models--" + MODEL.replace("/", "--")))


HOW = ("press \"Accept & start listening pass\" in the deck, or rerun with "
       "--accept-flamingo")
