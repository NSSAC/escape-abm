"""Helper utilities for reading output.h5 file."""

from pathlib import Path

import numpy as np
import polars as pl
import h5py as h5

import click
import rich

from .misc import RichException
from .parse_tree import mk_pt
from .ast import mk_ast, Source, Contagion
from .click_helpers import (
    simulation_file_option,
    existing_output_file_option,
    summary_file_option,
    transitions_file_option,
    transmissions_file_option,
    interventions_file_option,
    contagion_name_option,
)


def find_contagion(name: str, source: Source) -> Contagion:
    if not name:
        name = source.contagions[0].name

    contagion = [c for c in source.contagions if c.name == name]
    if len(contagion) != 1:
        rich.print("[red]Invalid contagion[/red]")
        raise SystemExit(1)
    contagion = contagion[0]

    return contagion


def save_df(df: pl.DataFrame, fname: Path):
    if fname.suffix == ".csv":
        df.write_csv(fname)
    elif fname.suffix == ".parquet":
        df.write_parquet(fname, compression="zstd")
    else:
        rich.print("[red]Unknown output file type[/red]")
        raise SystemExit(1)


def make_source(simulation_file: Path) -> Source:
    simulation_bytes = simulation_file.read_bytes()
    pt = mk_pt(str(simulation_file), simulation_bytes)
    ast = mk_ast(simulation_file, pt)
    return ast


def do_extract_summary(sim_output: h5.File, contagion: Contagion) -> pl.DataFrame:
    states = [const for const in contagion.state_type.value.consts]
    group = sim_output[contagion.name]["state_count"]  # type: ignore

    num_ticks = sim_output.attrs["num_ticks"]

    ticks = []
    counts = []
    for tick in range(num_ticks):  # type: ignore
        dset_name = f"tick_{tick}"
        dset = group[dset_name][...]  # type: ignore

        ticks.append(tick)
        counts.append(dset)
    ticks = np.array(ticks)
    counts = np.array(counts)

    df = pl.DataFrame(counts, states)
    df = df.with_columns(pl.Series(ticks).alias("tick"))
    return df


def read_summary_df(
    simulation_file: Path,
    sim_output_file: Path,
    contagion_name: str = "",
) -> pl.DataFrame:
    """Extract summary from simulation output."""
    ast1 = make_source(simulation_file)
    contagion = find_contagion(contagion_name, ast1)

    with h5.File(sim_output_file, "r") as sim_output:
        df = do_extract_summary(sim_output, contagion)
    return df


def do_extract_transitions(
    sim_output: h5.File,
    contagion: Contagion,
    gkey: str = "transitions",
) -> pl.DataFrame:
    states = [const for const in contagion.state_type.value.consts]
    states = pl.Enum(states)

    group = sim_output[contagion.name][gkey]  # type: ignore
    num_ticks = sim_output.attrs["num_ticks"]

    parts = []
    for tick in range(num_ticks):  # type: ignore
        tick_gname = f"tick_{tick}"
        tick_group = group[tick_gname]  # type: ignore

        node_index = tick_group["node_index"][...]  # type: ignore
        state = tick_group["state"][...]  # type: ignore
        part = {"node_index": node_index, "state": state}
        part = pl.DataFrame(part)
        part = part.with_columns(
            pl.lit(tick).alias("tick"), pl.col("state").cast(states)
        )
        parts.append(part)

    df = pl.concat(parts)
    return df


def do_extract_interventions(sim_output: h5.File, contagion: Contagion) -> pl.DataFrame:
    return do_extract_transitions(sim_output, contagion, "interventions")


def read_transitions_df(
    simulation_file: Path,
    sim_output_file: Path,
    contagion_name: str = "",
) -> pl.DataFrame:
    """Extract transitions from simulation output."""
    ast1 = make_source(simulation_file)
    contagion = find_contagion(contagion_name, ast1)

    with h5.File(sim_output_file, "r") as sim_output:
        df = do_extract_transitions(sim_output, contagion)
    return df


def read_interventions_df(
    simulation_file: Path,
    sim_output_file: Path,
    contagion_name: str = "",
) -> pl.DataFrame:
    """Extract interventions from simulation output."""
    ast1 = make_source(simulation_file)
    contagion = find_contagion(contagion_name, ast1)

    with h5.File(sim_output_file, "r") as sim_output:
        df = do_extract_interventions(sim_output, contagion)
    return df


def do_extract_transmissions(
    sim_output: h5.File,
    contagion: Contagion,
) -> pl.DataFrame:
    states = [const for const in contagion.state_type.value.consts]
    states = pl.Enum(states)

    group = sim_output[contagion.name]["transmissions"]  # type: ignore
    num_ticks = sim_output.attrs["num_ticks"]

    parts = []
    for tick in range(num_ticks):  # type: ignore
        tick_gname = f"tick_{tick}"
        tick_group = group[tick_gname]  # type: ignore

        edge_index = tick_group["edge_index"][...]  # type: ignore

        state = tick_group["state"][...]  # type: ignore
        part = {"edge_index": edge_index, "state": state}
        part = pl.DataFrame(part)
        part = part.with_columns(
            pl.lit(tick).alias("tick"), pl.col("state").cast(states)
        )

        parts.append(part)

    df = pl.concat(parts)
    return df


def read_transmissions_df(
    simulation_file: Path,
    sim_output_file: Path,
    contagion_name: str = "",
) -> pl.DataFrame:
    """Extract transmissions from simulation output."""
    ast1 = make_source(simulation_file)
    contagion = find_contagion(contagion_name, ast1)

    with h5.File(sim_output_file, "r") as sim_output:
        df = do_extract_transmissions(sim_output, contagion)
    return df


@click.group
def process_output():
    """Process output file."""


@process_output.command()
@simulation_file_option
@contagion_name_option
@existing_output_file_option
@summary_file_option
def extract_summary(
    simulation_file: Path,
    contagion_name: str,
    output_file: Path,
    summary_file: Path,
):
    """Extract summary from simulation output."""
    try:
        df = read_summary_df(simulation_file, output_file, contagion_name)
        save_df(df, summary_file)
    except RichException as e:
        e.rich_print()
        raise SystemExit(1)


@process_output.command()
@simulation_file_option
@contagion_name_option
@existing_output_file_option
@interventions_file_option
def extract_interventions(
    simulation_file: Path,
    contagion_name: str,
    output_file: Path,
    interventions_file: Path,
):
    """Extract interventions from simulation output."""
    try:
        df = read_interventions_df(simulation_file, output_file, contagion_name)
        save_df(df, interventions_file)
    except RichException as e:
        e.rich_print()
        raise SystemExit(1)


@process_output.command()
@simulation_file_option
@contagion_name_option
@existing_output_file_option
@transitions_file_option
def extract_transitions(
    simulation_file: Path,
    contagion_name: str,
    output_file: Path,
    transitions_file: Path,
):
    """Extract transitions from simulation output."""
    try:
        df = read_transitions_df(simulation_file, output_file, contagion_name)
        save_df(df, transitions_file)
    except RichException as e:
        e.rich_print()
        raise SystemExit(1)


@process_output.command()
@simulation_file_option
@contagion_name_option
@existing_output_file_option
@transmissions_file_option
def extract_transmissions(
    simulation_file: Path,
    contagion_name: str,
    output_file: Path,
    transmissions_file: Path,
):
    """Extract transmissions from simulation output."""
    try:
        df = read_transmissions_df(simulation_file, output_file, contagion_name)
        save_df(df, transmissions_file)
    except RichException as e:
        e.rich_print()
        raise SystemExit(1)


@process_output.command()
@simulation_file_option
@contagion_name_option
@existing_output_file_option
@summary_file_option
@interventions_file_option
@transitions_file_option
@transmissions_file_option
def extract_all(
    simulation_file: Path,
    contagion_name: str,
    output_file: Path,
    summary_file: Path,
    interventions_file: Path,
    transitions_file: Path,
    transmissions_file: Path,
):
    """Extract summary,interventions,transitions,transmissions from simulation output."""
    try:
        ast1 = make_source(simulation_file)
        contagion = find_contagion(contagion_name, ast1)

        with h5.File(output_file, "r") as sim_output:
            rich.print("[yellow]Extracting summary.[/yellow]")
            df = do_extract_summary(sim_output, contagion)
            save_df(df, summary_file)

            rich.print("[yellow]Extracting interventions.[/yellow]")
            df = do_extract_interventions(sim_output, contagion)
            save_df(df, interventions_file)

            rich.print("[yellow]Extracting transitions.[/yellow]")
            df = do_extract_transitions(sim_output, contagion)
            save_df(df, transitions_file)

            rich.print("[yellow]Extracting transmissions.[/yellow]")
            df = do_extract_transmissions(sim_output, contagion)
            save_df(df, transmissions_file)
    except RichException as e:
        e.rich_print()
        raise SystemExit(1)
