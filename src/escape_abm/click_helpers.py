"""Click helpers."""

from pathlib import Path

import click

ExistingFile = click.Path(exists=True, dir_okay=False, path_type=Path)
File = click.Path(dir_okay=False, path_type=Path)

ExistingDirectory = click.Path(exists=True, file_okay=False, path_type=Path)
Directory = click.Path(file_okay=False, path_type=Path)

simulation_file_option = click.option(
    "-s", "--simulation-file", type=ExistingFile, required=True, help="Simulation file."
)

input_file_option = click.option(
    "-i",
    "--input-file",
    type=File,
    default="input.h5",
    show_default=True,
    help="Simulation input file.",
)

existing_input_file_option = click.option(
    "-i",
    "--input-file",
    type=ExistingFile,
    default="input.h5",
    show_default=True,
    help="Simulation input file.",
)

output_file_option = click.option(
    "-o",
    "--output-file",
    type=File,
    default="output.h5",
    show_default=True,
    help="Simulation output file.",
)

existing_output_file_option = click.option(
    "-o",
    "--output-file",
    type=ExistingFile,
    default="output.h5",
    show_default=True,
    help="Simulation output file.",
)

gen_code_dir_option = click.option(
    "-g",
    "--generated-code-dir",
    "gen_code_dir",
    type=Directory,
    default=".",
    show_default=True,
    help="Generated code directory.",
)

existing_gen_code_dir_option = click.option(
    "-g",
    "--generated-code-dir",
    "gen_code_dir",
    type=ExistingDirectory,
    default=".",
    show_default=True,
    help="Generated code directory.",
)

node_file_option = click.option(
    "-n",
    "--node-file",
    type=File,
    required=True,
    help="Path to node attributes file.",
)

existing_node_file_option = click.option(
    "-n",
    "--node-file",
    type=ExistingFile,
    required=True,
    help="Path to node attributes file.",
)

edge_file_option = click.option(
    "-e",
    "--edge-file",
    type=File,
    required=True,
    help="Path to edge attributes file.",
)

existing_edge_file_option = click.option(
    "-e",
    "--edge-file",
    type=ExistingFile,
    required=True,
    help="Path to edge attributes file.",
)

contagion_name_option = click.option(
    "-c",
    "--contagion-name",
    default="",
    help="Contagion to extract. (default: first defined contagion)",
)

summary_file_option = click.option(
    "-so",
    "--summary-file",
    type=File,
    default="summary.csv",
    show_default=True,
    help="Path to summary file.",
)

interventions_file_option = click.option(
    "-io",
    "--interventions-file",
    type=File,
    default="interventions.csv",
    show_default=True,
    help="Path to interventions file.",
)

transitions_file_option = click.option(
    "-tio",
    "--transitions-file",
    type=File,
    default="transitions.csv",
    show_default=True,
    help="Path to transitions file.",
)

transmissions_file_option = click.option(
    "-tmo",
    "--transmissions-file",
    type=File,
    default="transmissions.csv",
    show_default=True,
    help="Path to transmissions file.",
)

statistics_file_option = click.option(
    "-sto",
    "--statistics-file",
    type=File,
    default="statistics.csv",
    show_default=True,
    help="Path to statistics file.",
)
