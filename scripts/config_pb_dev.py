"""Parsl config for PB's dev laptop."""

import sys
import sqlite3
from pathlib import Path
from subprocess import run

import parsl
from parsl.config import Config
from parsl.channels import LocalChannel
from parsl.providers import LocalProvider
from parsl.executors import HighThroughputExecutor
from parsl.monitoring import MonitoringHub
from parsl.addresses import address_by_interface

from parsl_helpers import make_fresh_dir

NUM_TASKS = 6
NUM_WORK_FILES = NUM_TASKS * 2
MPIRUN_COMMAND = f"/usr/bin/mpirun -n {NUM_TASKS}"
WORKER_CONDA_ENV = "synpop_make_network"

CURDIR = Path(__file__).parent
WORKER_INIT = f"""
. /home/parantapa/miniconda3/etc/profile.d/conda.sh
conda activate {WORKER_CONDA_ENV}

export PYTHONPATH='{CURDIR!s}'
"""


def setup_parsl(output_root: Path):
    parsl_work_dir = output_root / "parsl_root"
    make_fresh_dir(parsl_work_dir)

    script_dir = str(parsl_work_dir / "script_dir")
    htex_work_dir = str(parsl_work_dir / "htex_work_dir")
    worker_logdir_root = str(parsl_work_dir / "worker_logdir_root")
    run_dir = str(parsl_work_dir / "runinfo")

    serial_htex = HighThroughputExecutor(
        provider=LocalProvider(
            channel=LocalChannel(script_dir=script_dir),
            worker_init=WORKER_INIT,
        ),
        label="serial",
        address="127.0.0.1",
        working_dir=htex_work_dir,
        worker_logdir_root=worker_logdir_root,
    )

    parallel_htex = HighThroughputExecutor(
        provider=LocalProvider(
            channel=LocalChannel(script_dir=script_dir),
            worker_init=WORKER_INIT,
        ),
        label="parallel",
        address="127.0.0.1",
        working_dir=htex_work_dir,
        worker_logdir_root=worker_logdir_root,
        max_workers=1,
        cores_per_worker=1e-6,
    )

    log_endpoint = "sqlite:///" + str(parsl_work_dir / "monitoring.db")

    monitoring = MonitoringHub(
        hub_address=address_by_interface("lo"),
        hub_port=20355,
        monitoring_debug=False,
        resource_monitoring_interval=10,
        logging_endpoint=log_endpoint,
        logdir=str(parsl_work_dir / "monitoring_logs"),
    )

    config = Config(
        executors=[serial_htex, parallel_htex],
        monitoring=monitoring,
        run_dir=run_dir,
        max_idletime=30,
        strategy="htex_auto_scale",
        initialize_logging=True,
    )

    return parsl.load(config)


def less_outfile(output_root: Path, tid: int, stderr: bool):
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
        run(cmd, stdin=sys.stdin, stdout=sys.stdout)
    except KeyboardInterrupt:
        pass
