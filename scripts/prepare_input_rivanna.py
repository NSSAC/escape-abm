#!/usr/bin/env python
"""Prepare input data using smn tool."""

import click
import rich
from pathlib import Path

from prepare_input_common import do_prepare_input_cpu
from config_rivanna import setup_parsl, less_outfile, NUM_WORK_FILES, MPIRUN_COMMAND

INPUT_ROOT = Path("/scratch/pb5gj/episim37/example1")
NODE_FILES_GLOB = "ca/ca_nodes_sim/part.*.parquet"
EDGE_FILES_GLOB = "ca/ca_edges_sim/part.*.parquet"
SIMULATION_FILE = INPUT_ROOT / "example1.esl37"

OUTPUT_ROOT = INPUT_ROOT / "ca/sim_data"


@click.group()
def cli():
    pass


@cli.command()
def run():
    rich.print("Setting up Parsl")
    setup_parsl(OUTPUT_ROOT)

    do_prepare_input_cpu(
        mpirun_command=MPIRUN_COMMAND,
        num_work_files=NUM_WORK_FILES,
        input_root=INPUT_ROOT,
        output_root=OUTPUT_ROOT,
        simulation_file=SIMULATION_FILE,
        node_files_glob=NODE_FILES_GLOB,
        edge_files_glob=EDGE_FILES_GLOB,
        data_file=OUTPUT_ROOT / "input.h5",
    )


@cli.command()
@click.argument(
    "tid",
    type=int,
    required=True,
)
def outless(tid: int):
    less_outfile(OUTPUT_ROOT, tid, False)


@cli.command()
@click.argument(
    "tid",
    type=int,
    required=True,
)
def errless(tid: int):
    less_outfile(OUTPUT_ROOT, tid, True)


if __name__ == "__main__":
    cli()
