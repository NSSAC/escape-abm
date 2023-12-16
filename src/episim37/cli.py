"""Command line interface."""

import click

from .parse_tree import print_parse_tree
from .ast1 import print_ast1
from .codegen_cpu import codegen_cpu
from .language_server import language_server
from .prepare_input import prepare_cpu
from .process_output import process_output


@click.group()
def cli():
    """EpiSim37: Epidemic Simulations on Contact Networks"""


cli.add_command(print_parse_tree)
cli.add_command(print_ast1)
cli.add_command(codegen_cpu)
cli.add_command(language_server)
cli.add_command(prepare_cpu)
cli.add_command(process_output)
