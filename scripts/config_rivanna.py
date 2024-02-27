"""Parsl config running EpiSim37 on Rivanna."""

from pathlib import Path

import parsl
from parsl.config import Config
from parsl.channels import LocalChannel
from parsl.launchers import SingleNodeLauncher
from parsl.monitoring import MonitoringHub
from parsl.providers import SlurmProvider
from parsl.executors import HighThroughputExecutor
from parsl.addresses import address_by_interface

from parsl_helpers import make_fresh_dir

MAX_NODES = 20
PARITION = "bii"
ACCOUNT = "bii_nssac"
QOS = "bii-half"
WALLTIME = "24:00:00"
CPUS_PER_NODE = 37

POSTGRES_EXE = "/scratch/pb5gj/conda-envs/pg_env/bin/postgres"
OPTUNA_DASH_EXE = "/scratch/pb5gj/conda-envs/episim37/bin/optuna-dashboard"
PARSL_DASH_EXE = "/scratch/pb5gj/conda-envs/episim37/bin/parsl-visualize"

WORKER_CONDA_ENV = "episim37"
CONDA_INSTALL_DIR = "/home/pb5gj/miniconda3"
CURDIR = Path(__file__).parent

WORKER_INIT = f"""
module load gcc/11.4.0 openmpi/4.1.4
module load hdf5/1.12.2
module load cmake/3.28.1

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
    nodes_per_block = 1
    max_blocks = MAX_NODES
    launcher = SingleNodeLauncher()
    max_workers = 1
    cores_per_worker = CPUS_PER_NODE
    parallelism = 1.0

    htex = HighThroughputExecutor(
        provider=SlurmProvider(
            partition=PARITION,
            account=ACCOUNT,
            qos=QOS,
            channel=LocalChannel(script_dir=script_dir),
            nodes_per_block=nodes_per_block,
            cores_per_node=CPUS_PER_NODE,
            min_blocks=0,
            init_blocks=0,
            max_blocks=max_blocks,
            parallelism=parallelism,
            walltime=WALLTIME,
            worker_init=WORKER_INIT,
            exclusive=False,
            launcher=launcher,
        ),
        label=label,
        address=address_by_interface("ib0"),
        working_dir=htex_work_dir,
        worker_logdir_root=worker_logdir_root,
        max_workers=max_workers,
        cores_per_worker=cores_per_worker,
    )

    return htex


def setup_parsl(parsl_work_dir: str | Path):
    parsl_work_dir = Path(parsl_work_dir)
    make_fresh_dir(parsl_work_dir)

    default_htex = make_executor(parsl_work_dir=parsl_work_dir)
    log_endpoint = "sqlite:///" + str(parsl_work_dir / "monitoring.db")

    monitoring = MonitoringHub(
        hub_address=address_by_interface("ib0"),
        # hub_port=20355,
        monitoring_debug=False,
        resource_monitoring_interval=10,
        logging_endpoint=log_endpoint,
        logdir=str(parsl_work_dir / "monitoring_logs"),
    )

    config = Config(
        executors=[default_htex],
        monitoring=monitoring,
        run_dir=str(parsl_work_dir / "runinfo"),
        max_idletime=30,
        strategy="htex_auto_scale",
        initialize_logging=True,
    )

    return parsl.load(config)
