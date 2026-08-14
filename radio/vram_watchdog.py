#!/usr/bin/env python3
"""Release the Music3 server's VRAM once idle — same contract as H3's watchdog.
The 5080 cannot hold both models (each needs ~15.5 GB of 16), so an idle
server must never sit on the card.
"""
import json
import time
import urllib.request

HOST = "http://127.0.0.1:8189"
IDLE_SECS = 900   # release after 15 minutes idle
POLL_SECS = 20


def api(path, payload=None):
    body = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"} if payload is not None else {}
    req = urllib.request.Request(HOST + path, data=body, headers=headers)
    return urllib.request.urlopen(req, timeout=10)


def queue_busy():
    with api("/queue") as r:
        q = json.load(r)
    return bool(q.get("queue_running")) or bool(q.get("queue_pending"))


def main():
    last_active = time.time()
    released = True  # nothing loaded at startup

    while True:
        try:
            if queue_busy():
                last_active = time.time()
                released = False
            elif not released and (time.time() - last_active) > IDLE_SECS:
                api("/free", {"unload_models": True, "free_memory": True})
                released = True
                print(f"released VRAM after {IDLE_SECS}s idle", flush=True)
        except urllib.error.URLError:
            pass  # server restarting; retry next poll
        except Exception as e:
            print(f"watchdog error: {e!r}", flush=True)
        time.sleep(POLL_SECS)


if __name__ == "__main__":
    main()
