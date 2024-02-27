"""Run optuna dashboard locally."""

import shlex
import shutil
import socket
import atexit
import subprocess
from textwrap import dedent
from pathlib import Path


class OptunaDashboard:
    def __init__(
        self,
        storage: str,
        artifact_dir: Path | str,
        dashboard_exe: Path | str | None = None,
        port: int = 8080,
        delete_if_exists: bool = False,
        verbose: bool = False,
    ):
        artifact_dir = Path(artifact_dir)

        if dashboard_exe is None:
            dashboard_exe = shutil.which("optuna-dashboard")
            if dashboard_exe is None:
                raise RuntimeError("Unable to find optuna-dashboard executable.")
        dashboard_exe = Path(dashboard_exe)

        if delete_if_exists and artifact_dir.exists():
            shutil.rmtree(artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)

        # Start dashboard in the background
        print("Starting dashboard ...")
        cmd = f"""
        '{dashboard_exe!s}'
            --host 0.0.0.0
            --port {port}
            --artifact-dir '{artifact_dir!s}'
            '{storage}'
        """
        if verbose:
            cmd_str = dedent(cmd.strip())
            print(f"executing: {cmd_str}")
        cmd = shlex.split(cmd)

        with open(artifact_dir / "optuna-dashboard.log", "at") as fobj:
            self._proc: subprocess.Popen | None = subprocess.Popen(
                cmd,
                stdout=fobj,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
            )

        hostname = socket.gethostname()
        self.dashboard_url = f"http://{hostname}:{port}"
        print(f"Optuna dashboard url: {self.dashboard_url}")

        atexit.register(self.close)

    def close(self):
        if self._proc is not None:
            if self._proc.poll() is None:
                self._proc.terminate()
            self._proc = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        exc_type, exc_val, exc_tb = exc_type, exc_val, exc_tb
        self.close()
