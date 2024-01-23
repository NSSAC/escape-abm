"""Process output data."""

import sys
from pathlib import Path

import pandas as pd
import h5py as h5

import click
import rich

from .parse_tree import mk_pt, ParseTreeConstructionError
from .ast1 import mk_ast1, ASTConstructionError, Source, Contagion


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

    num_ticks = sim_output.attrs["num_ticks"]

    ticks = []
    counts = []
    for tick in range(-1, num_ticks):  # type: ignore
        dset_name = f"tick_{tick}"
        dset = group[dset_name][...]  # type: ignore

        ticks.append(tick)
        counts.append(dset)

    df = pd.DataFrame(counts, columns=states)
    df["tick"] = ticks
    return df


def do_extract_transitions(
    sim_input: h5.File,
    sim_output: h5.File,
    contagion: Contagion,
    node_key_dataset: str,
    gkey: str = "transitions",
) -> pd.DataFrame:
    states = [const for const in contagion.state_type.resolve().consts]
    states = {i: n for i, n in enumerate(states)}

    group = sim_output[contagion.name][gkey]  # type: ignore
    num_ticks = sim_output.attrs["num_ticks"]

    node_key = sim_input[node_key_dataset][...]  # type: ignore

    parts = []
    for tick in range(-1, num_ticks):  # type: ignore
        tick_gname = f"tick_{tick}"
        tick_group = group[tick_gname]  # type: ignore

        node_index = tick_group["node_index"][...]  # type: ignore
        node = node_key[node_index]  # type: ignore
        state = tick_group["state"][...]  # type: ignore
        part = {"node": node, "state": state}
        part = pd.DataFrame(part)
        part["tick"] = tick
        part["state"] = part.state.map(states)
        parts.append(part)

    df = pd.concat(parts, axis=0)
    return df


def do_extract_interventions(
    sim_input: h5.File, sim_output: h5.File, contagion: Contagion, node_key_dataset: str
) -> pd.DataFrame:
    return do_extract_transitions(
        sim_input, sim_output, contagion, node_key_dataset, "interventions"
    )


def do_extract_transmissions(
    sim_output: h5.File,
    contagion: Contagion,
) -> pd.DataFrame:
    states = [const for const in contagion.state_type.resolve().consts]
    states = {i: n for i, n in enumerate(states)}

    group = sim_output[contagion.name]["transmissions"]  # type: ignore
    num_ticks = sim_output.attrs["num_ticks"]

    parts = []
    for tick in range(-1, num_ticks):  # type: ignore
        tick_gname = f"tick_{tick}"
        tick_group = group[tick_gname]  # type: ignore

        edge_index = tick_group["edge_index"][...]  # type: ignore

        state = tick_group["state"][...]  # type: ignore
        part = {"edge_index": edge_index, "state": state}
        part = pd.DataFrame(part)
        part["tick"] = tick
        part["state"] = part.state.map(states)
        parts.append(part)

    df = pd.concat(parts, axis=0)
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

simulation_input_option = click.option(
    "-i",
    "--simulation-input",
    "sim_input_file",
    type=ExistingFile,
    required=True,
    show_default=True,
    help="Path to simulation input file (h5) file.",
)

simulation_output_option = click.option(
    "-o",
    "--simulation-output",
    "sim_output_file",
    type=ExistingFile,
    required=True,
    show_default=True,
    help="Path to simulation output file (h5) file.",
)

summary_file_option = click.option(
    "-so",
    "--summary-output",
    "summary_file",
    type=NewFile,
    default="summary.csv",
    show_default=True,
    help="Path to summary file.",
)

interventions_file_option = click.option(
    "-io",
    "--interventions-output",
    "interventions_file",
    type=NewFile,
    default="interventions.csv",
    show_default=True,
    help="Path to interventions file.",
)

transitions_file_option = click.option(
    "-tio",
    "--transitions-output",
    "transitions_file",
    type=NewFile,
    default="transitions.csv",
    show_default=True,
    help="Path to transitions file.",
)

transmissions_file_option = click.option(
    "-tmo",
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
@simulation_input_option
@simulation_output_option
@interventions_file_option
def extract_interventions(
    simulation_file: Path,
    contagion_name: str,
    sim_input_file: Path,
    sim_output_file: Path,
    interventions_file: Path,
):
    """Extract interventions from simulation output."""
    simulation_bytes = simulation_file.read_bytes()
    try:
        pt = mk_pt(str(simulation_file), simulation_bytes)
        ast1 = mk_ast1(simulation_file, pt)

        node_key_dataset = f"/node/{ast1.node_table.key.name}"

        contagion = find_contagion(contagion_name, ast1)
        with h5.File(sim_input_file, "r") as sim_input:
            with h5.File(sim_output_file, "r") as sim_output:
                df = do_extract_interventions(
                    sim_input, sim_output, contagion, node_key_dataset
                )

        save_df(df, interventions_file)
    except (ParseTreeConstructionError, ASTConstructionError) as e:
        e.rich_print()
        sys.exit(1)


@process_output.command()
@simulation_file_option
@contagion_name_option
@simulation_input_option
@simulation_output_option
@transitions_file_option
def extract_transitions(
    simulation_file: Path,
    contagion_name: str,
    sim_input_file: Path,
    sim_output_file: Path,
    transitions_file: Path,
):
    """Extract transitions from simulation output."""
    simulation_bytes = simulation_file.read_bytes()
    try:
        pt = mk_pt(str(simulation_file), simulation_bytes)
        ast1 = mk_ast1(simulation_file, pt)

        node_key_dataset = f"/node/{ast1.node_table.key.name}"

        contagion = find_contagion(contagion_name, ast1)
        with h5.File(sim_input_file, "r") as sim_input:
            with h5.File(sim_output_file, "r") as sim_output:
                df = do_extract_transitions(
                    sim_input, sim_output, contagion, node_key_dataset
                )

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
@simulation_input_option
@simulation_output_option
@summary_file_option
@interventions_file_option
@transitions_file_option
@transmissions_file_option
def extract_all(
    simulation_file: Path,
    contagion_name: str,
    sim_input_file: Path,
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

        node_key_dataset = f"/node/{ast1.node_table.key.name}"

        contagion = find_contagion(contagion_name, ast1)
        with h5.File(sim_input_file, "r") as sim_input:
            with h5.File(sim_output_file, "r") as sim_output:
                rich.print("[yellow]Extracting summary.[/yellow]")
                df = do_extract_summary(sim_output, contagion)
                save_df(df, summary_file)

                rich.print("[yellow]Extracting interventions.[/yellow]")
                df = do_extract_interventions(
                    sim_input, sim_output, contagion, node_key_dataset
                )
                save_df(df, interventions_file)

                rich.print("[yellow]Extracting transitions.[/yellow]")
                df = do_extract_transitions(
                    sim_input, sim_output, contagion, node_key_dataset
                )
                save_df(df, transitions_file)

                rich.print("[yellow]Extracting transmissions.[/yellow]")
                df = do_extract_transmissions(sim_output, contagion)
                save_df(df, transmissions_file)
    except (ParseTreeConstructionError, ASTConstructionError) as e:
        e.rich_print()
        sys.exit(1)
