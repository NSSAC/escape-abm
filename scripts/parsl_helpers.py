"""Parsl helper methods."""

import sys
import shutil
import sqlite3
import subprocess
from pathlib import Path


def make_fresh_dir(dir: Path) -> None:
    if dir.exists():
        shutil.rmtree(dir)
    dir.mkdir(mode=0o770, parents=True, exist_ok=False)


def outfile_pattern(
    task_executor: str, block_id: str, slurm_provider: bool, stderr: bool
) -> str:
    if slurm_provider:
        if stderr:
            return f"parsl.{task_executor}.block-{block_id}.*.*.submit.stderr"
        else:
            return f"parsl.{task_executor}.block-{block_id}.*.*.submit.stdout"
    else:
        if stderr:
            return f"parsl.{task_executor}.block-{block_id}.*.*.sh.err"
        else:
            return f"parsl.{task_executor}.block-{block_id}.*.*.sh.out"


def less_outfile(output_root: Path, tid: int, stderr: bool, slurm_provider: bool):
    """Find stderr file for tasks."""
    db = output_root / "parsl_root/monitoring.db"
    with sqlite3.connect(db) as con:
        sql = """
        select task_executor, block_id
        from try
        where task_id = ? and try_id = 0
        """
        cur = con.execute(sql, (tid,))
        rows = cur.fetchall()
        if len(rows) != 1:
            sys.exit(1)

        task_executor, block_id = rows[0]

    fname_pattern = outfile_pattern(task_executor, block_id, slurm_provider, stderr)

    if stderr:
        fname_pattern = f"parsl.{task_executor}.block-{block_id}.*.*.sh.err"
    else:
        fname_pattern = f"parsl.{task_executor}.block-{block_id}.*.*.sh.out"

    log_root = output_root / "parsl_root"
    outfiles = list(log_root.rglob(fname_pattern))

    if not outfiles:
        print("No matching files found.")
        return

    print("Matching files:")
    for file in outfiles:
        print(str(file))

    cmd = ["less", "-K", str(outfiles[0])]
    try:
        subprocess.run(cmd, stdin=sys.stdin, stdout=sys.stdout)
    except KeyboardInterrupt:
        pass
