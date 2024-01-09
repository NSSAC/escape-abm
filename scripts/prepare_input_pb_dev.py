#!/usr/bin/env python
"""Prepare input data using smn tool."""

import click
import rich
from pathlib import Path

from prepare_input_common import do_prepare_input_cpu
from config_pb_dev import setup_parsl, less_outfile, NUM_WORK_FILES, MPIRUN_COMMAND

INPUT_ROOT = Path("/home/parantapa/data/episim37-example1")
NODE_FILES_GLOB = "nodes.parquet"
EDGE_FILES_GLOB = "edges.parquet"
SIMULATION_FILE = INPUT_ROOT / "example1.esl37"

OUTPUT_ROOT = INPUT_ROOT / "input_par_temp"


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
        data_file=INPUT_ROOT / "input_par.h5",
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
