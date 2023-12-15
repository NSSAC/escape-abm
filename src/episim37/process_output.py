"""Process output data."""

import sys
from pathlib import Path

import pandas as pd
import h5py as h5

import click
import rich

from .ast1 import mk_ast1, ASTConstructionError, Source, Contagion
from .parse_tree import mk_pt, ParseTreeConstructionError


def find_contagion(name: str, source: Source) -> Contagion:
    if not name:
        name = source.contagions[0].name

    contagion = [c for c in source.contagions if c.name == name]
    if len(contagion) != 1:
        rich.print("[red]Invalid contagion[/red]")
        sys.exit(1)
    contagion = contagion[0]

    return contagion


def save_df(df: pd.DataFrame, fname: Path):
    if fname.suffix == ".csv":
        df.to_csv(fname, index=False)
    elif fname.suffix == ".parquet":
        df.to_parquet(fname, index=False, engine="pyarrow", compression="zstd")
    else:
        rich.print("[red]Unknown output file type[/red]")
        sys.exit(1)


def do_extract_summary(sim_output: h5.File, contagion: Contagion) -> pd.DataFrame:
    states = [const for const in contagion.state_type.resolve().consts]
    group = sim_output[contagion.name]["state_count"]  # type: ignore

    ticks = []
    counts = []
    for key in group.keys():  # type: ignore
        tick = key.split("_")
        tick = tick[-1]
        tick = int(tick)

        dset = group[key][...]  # type: ignore

        ticks.append(tick)
        counts.append(dset)

    df = pd.DataFrame(counts, columns=states)
    df["tick"] = ticks
    df.sort_values("tick", ignore_index=True, inplace=True)
    return df


def do_extract_transitions(
    sim_output: h5.File, contagion: Contagion, gkey: str = "transitions"
) -> pd.DataFrame:
    states = [const for const in contagion.state_type.resolve().consts]
    states = {i: n for i, n in enumerate(states)}

    group = sim_output[contagion.name][gkey]  # type: ignore

    parts = []
    for k1 in group.keys():  # type: ignore
        tick = k1.split("_")
        tick = tick[-1]
        tick = int(tick)

        for k2 in group[k1].keys():  # type: ignore
            node_index = group[k1][k2]["node_index"][...]  # type: ignore
            state = group[k1][k2]["state"][...]  # type: ignore
            part = {"node_index": node_index, "state": state}
            part = pd.DataFrame(part)
            part["tick"] = tick
            part["state"] = part.state.map(states)
            parts.append(part)

    df = pd.concat(parts, axis=0)
    df.sort_values(["tick", "node_index"], ignore_index=True, inplace=True)
    return df


def do_extract_interventions(sim_output: h5.File, contagion: Contagion) -> pd.DataFrame:
    return do_extract_transitions(sim_output, contagion, "interventions")


def do_extract_transmissions(sim_output: h5.File, contagion: Contagion) -> pd.DataFrame:
    states = [const for const in contagion.state_type.resolve().consts]
    states = {i: n for i, n in enumerate(states)}

    group = sim_output[contagion.name]["transmissions"]  # type: ignore

    parts = []
    for k1 in group.keys():  # type: ignore
        tick = k1.split("_")
        tick = tick[-1]
        tick = int(tick)

        for k2 in group[k1].keys():  # type: ignore
            node_index = group[k1][k2]["node_index"][...]  # type: ignore
            source_edge_index = group[k1][k2]["source_edge_index"][...]  # type: ignore
            state = group[k1][k2]["state"][...]  # type: ignore
            part = {
                "node_index": node_index,
                "source_edge_index": source_edge_index,
                "state": state,
            }
            part = pd.DataFrame(part)
            part["tick"] = tick
            part["state"] = part.state.map(states)
            parts.append(part)

    df = pd.concat(parts, axis=0)
    df.sort_values(["tick", "node_index"], ignore_index=True, inplace=True)
    return df


ExistingFile = click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path)
NewFile = click.Path(exists=False, file_okay=True, dir_okay=False, path_type=Path)


@click.group
def process_output():
    """Process output file."""


simulation_file_option = click.option(
    "-s",
    "--simulation",
    "simulation_file",
    type=ExistingFile,
    required=True,
    help="Path to simulation file.",
)

contagion_name_option = click.option(
    "-c",
    "--contagion",
    "contagion_name",
    default="",
    help="Contagion to extract. (default: first defined contagion)",
)

simulation_output_option = click.option(
    "-i",
    "--simulation-output",
    "sim_output_file",
    type=ExistingFile,
    required=True,
    show_default=True,
    help="Path to simulation output file (h5) file.",
)

summary_file_option = click.option(
    "-o",
    "--summary-output",
    "summary_file",
    type=NewFile,
    default="summary.csv",
    show_default=True,
    help="Path to summary file.",
)

interventions_file_option = click.option(
    "-o",
    "--interventions-output",
    "interventions_file",
    type=NewFile,
    default="interventions.csv",
    show_default=True,
    help="Path to interventions file.",
)

transitions_file_option = click.option(
    "-o",
    "--transitions-output",
    "transitions_file",
    type=NewFile,
    default="transitions.csv",
    show_default=True,
    help="Path to transitions file.",
)

transmissions_file_option = click.option(
    "-o",
    "--transmissions-output",
    "transmissions_file",
    type=NewFile,
    default="transmissions.csv",
    show_default=True,
    help="Path to transmissions file.",
)


@process_output.command()
@simulation_file_option
@contagion_name_option
@simulation_output_option
@summary_file_option
def extract_summary(
    simulation_file: Path,
    contagion_name: str,
    sim_output_file: Path,
    summary_file: Path,
):
    """Extract summary from simulation output."""
    simulation_bytes = simulation_file.read_bytes()
    try:
        pt = mk_pt(str(simulation_file), simulation_bytes)
        ast1 = mk_ast1(simulation_file, pt)

        contagion = find_contagion(contagion_name, ast1)
        with h5.File(sim_output_file, "r") as sim_output:
            df = do_extract_summary(sim_output, contagion)

        save_df(df, summary_file)
    except (ParseTreeConstructionError, ASTConstructionError) as e:
        e.rich_print()
        sys.exit(1)


@process_output.command()
@simulation_file_option
@contagion_name_option
@simulation_output_option
@interventions_file_option
def extract_interventions(
    simulation_file: Path,
    contagion_name: str,
    sim_output_file: Path,
    interventions_file: Path,
):
    """Extract interventions from simulation output."""
    simulation_bytes = simulation_file.read_bytes()
    try:
        pt = mk_pt(str(simulation_file), simulation_bytes)
        ast1 = mk_ast1(simulation_file, pt)

        contagion = find_contagion(contagion_name, ast1)
        with h5.File(sim_output_file, "r") as sim_output:
            df = do_extract_interventions(sim_output, contagion)

        save_df(df, interventions_file)
    except (ParseTreeConstructionError, ASTConstructionError) as e:
        e.rich_print()
        sys.exit(1)


@process_output.command()
@simulation_file_option
@contagion_name_option
@simulation_output_option
@transitions_file_option
def extract_transitions(
    simulation_file: Path,
    contagion_name: str,
    sim_output_file: Path,
    transitions_file: Path,
):
    """Extract transitions from simulation output."""
    simulation_bytes = simulation_file.read_bytes()
    try:
        pt = mk_pt(str(simulation_file), simulation_bytes)
        ast1 = mk_ast1(simulation_file, pt)

        contagion = find_contagion(contagion_name, ast1)
        with h5.File(sim_output_file, "r") as sim_output:
            df = do_extract_transitions(sim_output, contagion)

        save_df(df, transitions_file)
    except (ParseTreeConstructionError, ASTConstructionError) as e:
        e.rich_print()
        sys.exit(1)


@process_output.command()
@simulation_file_option
@contagion_name_option
@simulation_output_option
@transmissions_file_option
def extract_transmissions(
    simulation_file: Path,
    contagion_name: str,
    sim_output_file: Path,
    transmissions_file: Path,
):
    """Extract transmissions from simulation output."""
    simulation_bytes = simulation_file.read_bytes()
    try:
        pt = mk_pt(str(simulation_file), simulation_bytes)
        ast1 = mk_ast1(simulation_file, pt)

        contagion = find_contagion(contagion_name, ast1)
        with h5.File(sim_output_file, "r") as sim_output:
            df = do_extract_transmissions(sim_output, contagion)

        save_df(df, transmissions_file)
    except (ParseTreeConstructionError, ASTConstructionError) as e:
        e.rich_print()
        sys.exit(1)


@process_output.command()
@simulation_file_option
@contagion_name_option
@simulation_output_option
@summary_file_option
@interventions_file_option
@transitions_file_option
@transmissions_file_option
def extract_all(
    simulation_file: Path,
    contagion_name: str,
    sim_output_file: Path,
    summary_file: Path,
    interventions_file: Path,
    transitions_file: Path,
    transmissions_file: Path,
):
    """Extract summary,interventions,transitions,transmissions from simulation output."""
    simulation_bytes = simulation_file.read_bytes()
    try:
        pt = mk_pt(str(simulation_file), simulation_bytes)
        ast1 = mk_ast1(simulation_file, pt)

        contagion = find_contagion(contagion_name, ast1)
        with h5.File(sim_output_file, "r") as sim_output:
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
    except (ParseTreeConstructionError, ASTConstructionError) as e:
        e.rich_print()
        sys.exit(1)
