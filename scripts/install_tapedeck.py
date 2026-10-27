import os
import platform
import subprocess
import sys
import shutil
from pathlib import Path

class TapedeckInstaller:
    def __init__(self):
        self.os_type = platform.system()
        self.script_dir = Path(__file__).parent.resolve()
        self.repo_root = self.script_dir.parent.resolve()
        self.venv_path = self.repo_root / "venv"
        self.python_exe = self.venv_path / ("Scripts" if self.os_type == "Windows" else "bin") / "python"
        self.req_file = self.repo_root / "requirements.txt"

    def log(self, msg):
        print(f"[TAPEDECK] {msg}")

    def error(self, msg):
        print(f"[ERROR] {msg}")
        sys.exit(1)

    def run_cmd(self, cmd, cwd=None, env=None):
        try:
            subprocess.check_call(cmd, shell=True, cwd=cwd, env=env)
        except subprocess.CalledProcessError as e:
            self.error(f"Command failed: {cmd}\n{e}")

    def check_dependencies(self):
        self.log("Checking system dependencies...")
        if self.os_type == "Linux":
            if shutil.which("ffmpeg") is None:
                self.log("Warning: ffmpeg not found.")
        elif self.os_type == "Windows":
            self.log("Note: For full GPU support, please run this inside WSL2.")
        elif self.os_type == "Darwin":
            self.log("Note: macOS is not officially supported.")

    def setup_venv(self):
        self.log(f"Creating virtual environment at {self.venv_path}...")
        if self.venv_path.exists():
            self.log("Existing venv found. Skipping creation.")
        else:
            self.run_cmd(f"{sys.executable} -m venv {self.venv_path}")

    def install_python_deps(self):
        if not self.req_file.exists():
            self.error(f"Requirements file not found at {self.req_file}")
        self.log(f"Installing Python dependencies from {self.req_file}...")
        self.run_cmd(f"{self.python_exe} -m pip install --upgrade pip")
        self.run_cmd(f"{self.python_exe} -m pip install -r {self.req_file}")

    def setup_macos_special(self):
        if self.os_type == "Darwin":
            self.log("Attempting to install torch with Metal support for macOS...")
            self.run_cmd(f"{self.python_exe} -m pip install torch torchvision torchaudio")

    def run(self):
        self.log(f"Starting installation for {self.os_type}...")
        self.log(f"Repo Root: {self.repo_root}")
        self.check_dependencies()
        self.setup_venv()
        if self.os_type == "Darwin":
            self.setup_macos_special()
        self.install_python_deps()
        self.log("Installation complete!")
        self.log(f"To start the radio, run: {self.python_exe} radio/tank_daemon.py")

if __name__ == "__main__":
    installer = TapedeckInstaller()
    installer.run()
