#!/usr/bin/env python3
"""∞ TAPEDECK tray: the one place the radio is started, watched and stopped.

The tray owns the whole stack while it runs:
  - ComfyUI (the deck + Music3 generation) starts with the tray.
  - The radio (radio/tank_daemon.py) runs only while "Radio on" is ticked.
    Off writes radio/DISABLED, so nothing else can start it behind the tray.
  - Captioning (the import pipeline's listening pass) pauses and resumes
    through the deck's own /music_studio/import/{pause,resume} routes;
    pausing ends the captioner, so its VRAM really comes back.
  - Quit pauses captioning (resumable), stops the radio, interrupts and frees
    ComfyUI, then stops it: nothing of the tapedeck is left on the GPU.

Processes keep the same run/*.pid files as bin/tapedeck, so `tapedeck status`
still reads true. Runs on a system Python with PySide6 (QtWebEngine for the
deck window): /usr/bin/python3 tray/tapedeck_tray.py
"""
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request

from PySide6.QtCore import QObject, QTimer, QUrl, Signal, Qt
from PySide6.QtGui import QAction, QIcon
from PySide6.QtNetwork import QLocalServer, QLocalSocket
from PySide6.QtWidgets import QApplication, QMainWindow, QMenu, QSystemTrayIcon

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def load_env():
    env = {"TAPEDECK": REPO, "COMFY": os.path.expanduser("~/tapedeck-comfy"),
           "PORT": "8188"}
    env["PYBIN"] = os.path.join(env["COMFY"], "venv/bin/python")
    try:
        for line in open(os.path.join(REPO, "run/tapedeck.env")):
            k, _, v = line.strip().partition("=")
            if k and v:
                env[k] = v.strip('"\'')
    except OSError:
        pass
    return env


ENV = load_env()
RUN = os.path.join(REPO, "run")
PORT = int(ENV["PORT"])
BASE = f"http://127.0.0.1:{PORT}"
DECK_URL = f"{BASE}/extensions/music_studio/index.html"
DISABLED = os.path.join(REPO, "radio", "DISABLED")
DATA = os.path.expanduser("~/.local/share/tapedeck")
SOCKET_NAME = f"tapedeck-tray-{os.getuid()}"


# ---------------------------------------------------------------- processes

def pid_of(name):
    try:
        pid = int(open(os.path.join(RUN, f"{name}.pid")).read().strip())
        os.kill(pid, 0)
        return pid
    except (OSError, ValueError):
        return None


def spawn(name, argv, cwd):
    os.makedirs(RUN, exist_ok=True)
    log = open(os.path.join(RUN, f"{name}.log"), "ab")
    p = subprocess.Popen(argv, cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, start_new_session=True)
    with open(os.path.join(RUN, f"{name}.pid"), "w") as f:
        f.write(str(p.pid))
    return p


def stop_pid(name, grace=20):
    """SIGTERM by saved PID, wait, SIGKILL only if it will not go."""
    pid = pid_of(name)
    if pid:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
        end = time.time() + grace
        while time.time() < end:
            try:
                os.kill(pid, 0)
            except OSError:
                break
            try:  # reap if it is our child
                os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                pass
            time.sleep(0.3)
        else:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
    try:
        os.unlink(os.path.join(RUN, f"{name}.pid"))
    except OSError:
        pass


def http(path, body=None, timeout=3):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(BASE + path, data=data,
                                 method="GET" if body is None else "POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    try:
        return json.loads(raw)
    except ValueError:
        return {}


def comfy_up():
    try:
        http("/music_studio/status", timeout=1.5)
        return True
    except Exception:
        return False


def start_comfy():
    if pid_of("comfy"):
        return
    spawn("comfy", [ENV["PYBIN"], "main.py", "--listen", "0.0.0.0",
                    "--port", str(PORT)], ENV["COMFY"])


def radio_wanted():
    return not os.path.exists(DISABLED)


def start_radio():
    try:
        os.unlink(DISABLED)
    except OSError:
        pass
    if not pid_of("daemon"):
        spawn("daemon", [ENV["PYBIN"], "radio/tank_daemon.py"], REPO)


def stop_radio():
    with open(DISABLED, "w") as f:
        f.write("radio switched off from the tray; tick Radio on to start it\n")
    stop_pid("daemon")


def free_comfy():
    for path, body in (("/interrupt", {}),
                       ("/free", {"unload_models": True, "free_memory": True})):
        try:
            http(path, body, timeout=5)
        except Exception:
            pass


# ---------------------------------------------------------------- the deck window

class Deck(QMainWindow):
    def __init__(self):
        super().__init__()
        from PySide6.QtWebEngineCore import QWebEnginePage, QWebEngineProfile, QWebEngineSettings
        from PySide6.QtWebEngineWidgets import QWebEngineView
        self.setWindowTitle("∞ TAPEDECK")
        self.resize(1280, 860)
        prof = QWebEngineProfile("tapedeck", self)  # persistent: keeps deck localStorage
        prof.setPersistentStoragePath(os.path.join(DATA, "webprofile"))
        prof.setCachePath(os.path.join(DATA, "webprofile-cache"))
        self.view = QWebEngineView(self)
        self.view.setPage(QWebEnginePage(prof, self.view))
        self.view.settings().setAttribute(
            QWebEngineSettings.WebAttribute.PlaybackRequiresUserGesture, False)
        self.setCentralWidget(self.view)
        self.loaded = False

    def show_deck(self):
        if not self.loaded:
            self.view.load(QUrl(DECK_URL))
            self.loaded = True
        self.showNormal()
        self.raise_()
        self.activateWindow()

    def closeEvent(self, ev):
        # Closing the window only hides it: the music keeps playing.
        # Quit in the tray menu is what ends everything.
        ev.ignore()
        self.hide()

    def unload(self):
        self.view.setUrl(QUrl("about:blank"))


# ---------------------------------------------------------------- the tray

class Status(QObject):
    changed = Signal(dict)
    done = Signal()        # worker threads cannot start Qt timers: signal instead


class Tray(QObject):
    def __init__(self, app):
        super().__init__()
        self.app = app
        self.deck = None
        self.quitting = False
        self.status = {"comfy": "starting", "import": {}}

        icon = QIcon(os.path.join(HERE, "tapedeck.svg"))
        app.setWindowIcon(icon)
        self.tray = QSystemTrayIcon(icon, self)
        self.tray.setToolTip("∞ TAPEDECK")

        m = QMenu()
        self.a_open = m.addAction("Open deck", self.open_deck)
        m.addSeparator()
        self.a_state = m.addAction("Starting…")
        self.a_state.setEnabled(False)
        self.a_caption = m.addAction("Pause captioning", self.toggle_caption)
        self.a_radio = QAction("Radio on", m, checkable=True)
        self.a_radio.setChecked(radio_wanted())
        self.a_radio.toggled.connect(self.toggle_radio)
        m.addAction(self.a_radio)
        m.addSeparator()
        m.addAction("Quit (free the GPU)", self.quit)
        self.menu = m
        self.tray.setContextMenu(m)
        self.tray.activated.connect(self.on_click)
        self.tray.show()

        self.sig = Status()
        self.sig.changed.connect(self.render)
        self.sig.done.connect(app.quit, Qt.ConnectionType.QueuedConnection)
        threading.Thread(target=self.boot, daemon=True).start()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.poll)
        self.timer.start(3000)

    # -- startup / polling, off the GUI thread
    def boot(self):
        start_comfy()
        for _ in range(240):
            if comfy_up() or self.quitting:
                break
            if not pid_of("comfy"):
                self.sig.changed.emit({"comfy": "failed", "import": {}})
                return
            time.sleep(1)
        if radio_wanted() and not self.quitting:
            start_radio()

    def poll(self):
        if self.quitting:
            return
        threading.Thread(target=self._poll, daemon=True).start()

    def _poll(self):
        if not comfy_up():
            st = "starting" if pid_of("comfy") else "stopped"
            self.sig.changed.emit({"comfy": st, "import": {}})
            return
        try:
            imp = http("/music_studio/import/status", timeout=2)
        except Exception:
            imp = {}
        self.sig.changed.emit({"comfy": "up", "import": imp})

    def render(self, s):
        if self.quitting:
            return
        self.status = s
        imp = s.get("import") or {}
        state = imp.get("state", "idle")
        if s["comfy"] != "up":
            self.a_state.setText({"starting": "Starting the deck…",
                                  "failed": "Deck failed to start (run/comfy.log)",
                                  }.get(s["comfy"], "Deck stopped"))
            self.a_caption.setEnabled(False)
        else:
            stages = imp.get("stages") or []
            i = imp.get("stage_idx", -1)
            stage = stages[i]["name"] if isinstance(i, int) and 0 <= i < len(stages) else ""
            if state == "running":
                txt = f"Capturing: {stage} {imp.get('overall_pct', 0)}%"
                if imp.get("track_n"):
                    txt += f" ({imp.get('track_i', 0)}/{imp.get('track_n')})"
            elif state in ("paused", "cancelled", "died", "failed"):
                txt = f"Captioning paused ({state})" if state != "paused" else "Captioning paused"
            elif state == "done":
                txt = "Captioning done"
            else:
                txt = "Captioning idle"
            self.a_state.setText(txt)
            if state == "running":
                self.a_caption.setText("Pause captioning")
                self.a_caption.setEnabled(True)
            elif state in ("paused", "cancelled", "died", "failed"):
                self.a_caption.setText("Resume captioning")
                self.a_caption.setEnabled(True)
            else:
                self.a_caption.setText("Pause captioning")
                self.a_caption.setEnabled(False)
        # keep the tick honest if something else flipped the switch
        want = radio_wanted()
        if self.a_radio.isChecked() != want:
            self.a_radio.blockSignals(True)
            self.a_radio.setChecked(want)
            self.a_radio.blockSignals(False)
        self.tray.setToolTip("∞ TAPEDECK — " + self.a_state.text())

    # -- actions
    def on_click(self, reason):
        if reason == QSystemTrayIcon.ActivationReason.Trigger:
            self.open_deck()

    def open_deck(self):
        if self.deck is None:
            self.deck = Deck()
        self.deck.show_deck()

    def toggle_caption(self):
        state = (self.status.get("import") or {}).get("state")
        path = "/music_studio/import/pause" if state == "running" else "/music_studio/import/resume"
        self.a_caption.setEnabled(False)
        self.a_state.setText("Pausing captioning…" if state == "running" else "Resuming captioning…")

        def go():
            try:
                http(path, {}, timeout=30)
            except Exception as e:
                print("caption toggle:", e, file=sys.stderr)
            self._poll()
        threading.Thread(target=go, daemon=True).start()

    def toggle_radio(self, on):
        threading.Thread(target=start_radio if on else stop_radio, daemon=True).start()

    def quit(self):
        if self.quitting:
            return
        self.quitting = True
        self.timer.stop()
        self.tray.setToolTip("∞ TAPEDECK — shutting down…")
        self.a_state.setText("Shutting down…")
        for a in self.menu.actions():
            a.setEnabled(False)
        if self.deck is not None:
            self.deck.unload()
            self.deck.hide()

        def go():
            if comfy_up():
                try:  # resumable stop of the captioner; it waits for teardown
                    if (http("/music_studio/import/status", timeout=2) or {}).get("state") == "running":
                        http("/music_studio/import/pause", {}, timeout=30)
                except Exception:
                    pass
            stop_pid("daemon")      # radio: stop, but leave the switch as it was
            free_comfy()
            stop_pid("comfy", grace=30)
            self.sig.done.emit()
        threading.Thread(target=go, daemon=True).start()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("tapedeck")
    app.setDesktopFileName("tapedeck")
    app.setQuitOnLastWindowClosed(False)

    # one tray only: a second launch just opens the deck in the first
    sock = QLocalSocket()
    sock.connectToServer(SOCKET_NAME)
    if sock.waitForConnected(500):
        sock.write(b"open")
        sock.waitForBytesWritten(500)
        return 0
    QLocalServer.removeServer(SOCKET_NAME)
    server = QLocalServer()
    server.listen(SOCKET_NAME)

    if not QSystemTrayIcon.isSystemTrayAvailable():
        print("no system tray available", file=sys.stderr)
    tray = Tray(app)

    def on_conn():
        c = server.nextPendingConnection()
        c.waitForReadyRead(500)
        if bytes(c.readAll()).strip() == b"open":
            tray.open_deck()
        c.close()
    server.newConnection.connect(on_conn)

    for s in (signal.SIGTERM, signal.SIGINT):  # a logout or Ctrl-C quits cleanly too
        signal.signal(s, lambda *_: QTimer.singleShot(0, tray.quit))
    tick = QTimer()  # let Python see signals while Qt sleeps
    tick.timeout.connect(lambda: None)
    tick.start(500)

    if "--open" in sys.argv:
        QTimer.singleShot(0, tray.open_deck)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
