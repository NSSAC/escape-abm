"""Run parsl dashboard locally."""

import shlex
import shutil
import atexit
import socket
import subprocess
from textwrap import dedent
from pathlib import Path


class ParslDashboard:
    def __init__(
        self,
        parsl_work_dir: Path | str,
        dashboard_exe: Path | str | None = None,
        port: int = 8081,
        verbose: bool = False,
    ):
        parsl_work_dir = Path(parsl_work_dir)
        log_path = parsl_work_dir / "parsl-visualize.log"

        if dashboard_exe is None:
            dashboard_exe = shutil.which("parsl-visualize")
            if dashboard_exe is None:
                raise RuntimeError("Unable to find parsl-visualize executable.")
        dashboard_exe = Path(dashboard_exe)

        log_endpoint = "sqlite:///" + str(parsl_work_dir / "monitoring.db")

        # Start dashboard in the background
        print("Starting dashboard ...")
        cmd = f"""
        '{dashboard_exe!s}'
            --l 0.0.0.0
            --port {port}
            '{log_endpoint!s}'
        """
        if verbose:
            cmd_str = dedent(cmd.strip())
            print(f"executing: {cmd_str}")
        cmd = shlex.split(cmd)

        with open(log_path, "at") as fobj:
            self._proc: subprocess.Popen | None = subprocess.Popen(
                cmd,
                stdout=fobj,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
            )

        hostname = socket.gethostname()
        self.dashboard_url = f"http://{hostname}:{port}"
        print(f"Parsl dashboard url: {self.dashboard_url}")

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
