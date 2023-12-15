"""Process output data."""

import sys
from pathlib import Path

import pandas as pd
import h5py as h5

import click
import rich

from .ast1 import mk_ast1, ASTConstructionError
from .parse_tree import mk_pt, ParseTreeConstructionError


@click.group
def process_output():
    """Process output file."""


@process_output.command()
@click.option(
    "-s",
    "--simulation",
    "simulation_file",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to simulation (esl37) file.",
)
@click.option(
    "-c",
    "--contagion",
    "contagion_name",
    default="",
    help="Contagion to extract. (default: first defined contagion)",
)
@click.option(
    "-i",
    "--simulation-output",
    "sim_output_file",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=True,
    show_default=True,
    help="Path to simulation output file (h5) file.",
)
@click.option(
    "-o",
    "--summary-output",
    "summary_file",
    type=click.Path(exists=False, file_okay=True, dir_okay=False, path_type=Path),
    default="summary.parquet",
    show_default=True,
    help="Path to summary output file.",
)
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

        if not contagion_name:
            contagion_name = ast1.contagions[0].name

        contagion = [c for c in ast1.contagions if c.name == contagion_name]
        if len(contagion) != 1:
            rich.print("[red]Invalid contagion[/red]")
            sys.exit(1)
        contagion = contagion[0]

        states = [const for const in contagion.state_type.resolve().consts]

        with h5.File(sim_output_file, "r") as sim_output:
            group = sim_output[contagion_name]["state_count"]  # type: ignore

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
        df.sort_values("tick", ignore_index=True)

        if summary_file.suffix == ".csv":
            df.to_csv(summary_file, index=False)
        elif summary_file.suffix == ".parquet":
            df.to_parquet(summary_file, index=False, engine="pyarrow")
        else:
            rich.print("[red]Unknown output file type[/red]")
            sys.exit(1)
    except (ParseTreeConstructionError, ASTConstructionError) as e:
        e.rich_print()
        sys.exit(1)


@process_output.command()
@click.option(
    "-s",
    "--simulation",
    "simulation_file",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to simulation (esl37) file.",
)
@click.option(
    "-c",
    "--contagion",
    "contagion_name",
    default="",
    help="Contagion to extract. (default: first defined contagion)",
)
@click.option(
    "-i",
    "--simulation-output",
    "sim_output_file",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=True,
    show_default=True,
    help="Path to simulation output file (h5) file.",
)
@click.option(
    "-o",
    "--summary-output",
    "summary_file",
    type=click.Path(exists=False, file_okay=True, dir_okay=False, path_type=Path),
    default="summary.parquet",
    show_default=True,
    help="Path to summary output file.",
)
def extract_interventions(
    simulation_file: Path,
    contagion_name: str,
    sim_output_file: Path,
    summary_file: Path,
):
    """Extract interventions from simulation output."""
    simulation_bytes = simulation_file.read_bytes()

    try:
        pt = mk_pt(str(simulation_file), simulation_bytes)
        ast1 = mk_ast1(simulation_file, pt)

        if not contagion_name:
            contagion_name = ast1.contagions[0].name

        contagion = [c for c in ast1.contagions if c.name == contagion_name]
        if len(contagion) != 1:
            rich.print("[red]Invalid contagion[/red]")
            sys.exit(1)
        contagion = contagion[0]

        states = [const for const in contagion.state_type.resolve().consts]
        states = {i: n for i, n in enumerate(states)}

        with h5.File(sim_output_file, "r") as sim_output:
            group = sim_output[contagion_name]["interventions"]  # type: ignore

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
        df.sort_values(["tick", "node_index"], ignore_index=True)

        if summary_file.suffix == ".csv":
            df.to_csv(summary_file, index=False)
        elif summary_file.suffix == ".parquet":
            df.to_parquet(summary_file, index=False, engine="pyarrow")
        else:
            rich.print("[red]Unknown output file type[/red]")
            sys.exit(1)
    except (ParseTreeConstructionError, ASTConstructionError) as e:
        e.rich_print()
        sys.exit(1)


@process_output.command()
@click.option(
    "-s",
    "--simulation",
    "simulation_file",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to simulation (esl37) file.",
)
@click.option(
    "-c",
    "--contagion",
    "contagion_name",
    default="",
    help="Contagion to extract. (default: first defined contagion)",
)
@click.option(
    "-i",
    "--simulation-output",
    "sim_output_file",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=True,
    show_default=True,
    help="Path to simulation output file (h5) file.",
)
@click.option(
    "-o",
    "--summary-output",
    "summary_file",
    type=click.Path(exists=False, file_okay=True, dir_okay=False, path_type=Path),
    default="summary.parquet",
    show_default=True,
    help="Path to summary output file.",
)
def extract_transitions(
    simulation_file: Path,
    contagion_name: str,
    sim_output_file: Path,
    summary_file: Path,
):
    """Extract transitions from simulation output."""
    simulation_bytes = simulation_file.read_bytes()

    try:
        pt = mk_pt(str(simulation_file), simulation_bytes)
        ast1 = mk_ast1(simulation_file, pt)

        if not contagion_name:
            contagion_name = ast1.contagions[0].name

        contagion = [c for c in ast1.contagions if c.name == contagion_name]
        if len(contagion) != 1:
            rich.print("[red]Invalid contagion[/red]")
            sys.exit(1)
        contagion = contagion[0]

        states = [const for const in contagion.state_type.resolve().consts]
        states = {i: n for i, n in enumerate(states)}

        with h5.File(sim_output_file, "r") as sim_output:
            group = sim_output[contagion_name]["transitions"]  # type: ignore

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
        df.sort_values(["tick", "node_index"], ignore_index=True)

        if summary_file.suffix == ".csv":
            df.to_csv(summary_file, index=False)
        elif summary_file.suffix == ".parquet":
            df.to_parquet(summary_file, index=False, engine="pyarrow")
        else:
            rich.print("[red]Unknown output file type[/red]")
            sys.exit(1)
    except (ParseTreeConstructionError, ASTConstructionError) as e:
        e.rich_print()
        sys.exit(1)


@process_output.command()
@click.option(
    "-s",
    "--simulation",
    "simulation_file",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to simulation (esl37) file.",
)
@click.option(
    "-c",
    "--contagion",
    "contagion_name",
    default="",
    help="Contagion to extract. (default: first defined contagion)",
)
@click.option(
    "-i",
    "--simulation-output",
    "sim_output_file",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=True,
    show_default=True,
    help="Path to simulation output file (h5) file.",
)
@click.option(
    "-o",
    "--summary-output",
    "summary_file",
    type=click.Path(exists=False, file_okay=True, dir_okay=False, path_type=Path),
    default="summary.parquet",
    show_default=True,
    help="Path to summary output file.",
)
def extract_transmissions(
    simulation_file: Path,
    contagion_name: str,
    sim_output_file: Path,
    summary_file: Path,
):
    """Extract transitions from simulation output."""
    simulation_bytes = simulation_file.read_bytes()

    try:
        pt = mk_pt(str(simulation_file), simulation_bytes)
        ast1 = mk_ast1(simulation_file, pt)

        if not contagion_name:
            contagion_name = ast1.contagions[0].name

        contagion = [c for c in ast1.contagions if c.name == contagion_name]
        if len(contagion) != 1:
            rich.print("[red]Invalid contagion[/red]")
            sys.exit(1)
        contagion = contagion[0]

        states = [const for const in contagion.state_type.resolve().consts]
        states = {i: n for i, n in enumerate(states)}

        with h5.File(sim_output_file, "r") as sim_output:
            group = sim_output[contagion_name]["transmissions"]  # type: ignore

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
        df.sort_values(["tick", "node_index"], ignore_index=True)

        if summary_file.suffix == ".csv":
            df.to_csv(summary_file, index=False)
        elif summary_file.suffix == ".parquet":
            df.to_parquet(summary_file, index=False, engine="pyarrow")
        else:
            rich.print("[red]Unknown output file type[/red]")
            sys.exit(1)
    except (ParseTreeConstructionError, ASTConstructionError) as e:
        e.rich_print()
        sys.exit(1)
