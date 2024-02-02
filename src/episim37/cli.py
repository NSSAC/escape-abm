"""Command line interface."""

import click

from .parse_tree import print_parse_tree
from .ast1 import print_ast1
from .codegen_cpu import codegen_cpu, print_cpu_ir
from .language_server import language_server
from .prepare_input import prepare_cpu
from .process_output import process_output


@click.group()
def cli():
    """EpiSim37: Epidemic Simulations on Contact Networks"""


@cli.group()
def debug():
    """Debug helpers"""


debug.add_command(print_parse_tree)
debug.add_command(print_ast1)
debug.add_command(print_cpu_ir)

cli.add_command(language_server)

cli.add_command(codegen_cpu)
cli.add_command(prepare_cpu)
cli.add_command(process_output)
