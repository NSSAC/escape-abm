"""Command line interface."""

import click

from .parse_tree import print_parse_tree

@click.group()
def cli():
    """EpiSim37: Epidemic Simulations on Contact Networks"""

cli.add_command(print_parse_tree)
