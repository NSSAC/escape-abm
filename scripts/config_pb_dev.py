"""Parsl config running EpiSim37 on PB's dev laptop."""

from pathlib import Path

import parsl
from parsl.config import Config
from parsl.channels import LocalChannel
from parsl.providers import LocalProvider
from parsl.executors import HighThroughputExecutor
from parsl.monitoring import MonitoringHub
from parsl.addresses import address_by_interface

from parsl_helpers import make_fresh_dir

CPU_CORES = 6
NUM_NODES = 1

POSTGRES_EXE = "/usr/bin/postgres"
OPTUNA_DASH_EXE = "/home/parantapa/miniconda3/envs/episim37/bin/optuna-dashboard"
PARSL_DASH_EXE = "/home/parantapa/miniconda3/envs/episim37/bin/parsl-visualize"

WORKER_CONDA_ENV = "episim37"
CONDA_INSTALL_DIR = "/home/parantapa/miniconda3"
CURDIR = Path(__file__).parent

WORKER_INIT = f"""
. "{CONDA_INSTALL_DIR}/etc/profile.d/conda.sh"
conda init
conda activate {WORKER_CONDA_ENV}

export PYTHONPATH='{CURDIR!s}':$PYTHONPATH
"""


def make_executor(parsl_work_dir: Path) -> HighThroughputExecutor:
    script_dir = str(parsl_work_dir / "script_dir")
    htex_work_dir = str(parsl_work_dir / "htex_work_dir")
    worker_logdir_root = str(parsl_work_dir / "worker_logdir_root")

    label = "default_htex"
    max_workers = 1

    htex = HighThroughputExecutor(
        provider=LocalProvider(
            channel=LocalChannel(script_dir=script_dir),
            worker_init=WORKER_INIT,
        ),
        label=label,
        address="127.0.0.1",
        working_dir=htex_work_dir,
        worker_logdir_root=worker_logdir_root,
        max_workers=max_workers,
        cores_per_worker=CPU_CORES,
    )

    return htex


def setup_parsl(parsl_work_dir: str | Path):
    parsl_work_dir = Path(parsl_work_dir)
    make_fresh_dir(parsl_work_dir)

    run_dir = str(parsl_work_dir / "runinfo")

    default_htex = make_executor(parsl_work_dir=parsl_work_dir)
    log_endpoint = "sqlite:///" + str(parsl_work_dir / "monitoring.db")

    monitoring = MonitoringHub(
        hub_address=address_by_interface("lo"),
        # hub_port=20355,
        monitoring_debug=False,
        resource_monitoring_interval=10,
        logging_endpoint=log_endpoint,
        logdir=str(parsl_work_dir / "monitoring_logs"),
    )

    config = Config(
        executors=[default_htex],
        monitoring=monitoring,
        run_dir=run_dir,
        max_idletime=30,
        strategy="htex_auto_scale",
        initialize_logging=True,
    )

    return parsl.load(config)
