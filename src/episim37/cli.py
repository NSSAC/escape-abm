"""Command line interface."""

import click

from .parse_tree import print_parse_tree
from .ast1 import print_ast1
from .ir1 import print_ir1
from .codegen_cpu import codegen_cpu


@click.group()
def cli():
    """EpiSim37: Epidemic Simulations on Contact Networks"""


cli.add_command(print_parse_tree)
cli.add_command(print_ast1)
cli.add_command(print_ir1)
cli.add_command(codegen_cpu)
