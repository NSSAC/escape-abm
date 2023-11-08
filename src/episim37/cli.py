"""Command line interface."""

import click

from .parse_tree import print_parse_tree
from .ast1 import print_ast1, print_ast2
# from .codegen_cpu import cpu


@click.group()
def cli():
    """EpiSim37: Epidemic Simulations on Contact Networks"""


cli.add_command(print_parse_tree)
cli.add_command(print_ast1)
cli.add_command(print_ast2)
# cli.add_command(cpu)
