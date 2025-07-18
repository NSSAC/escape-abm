"""Check for types."""

from pathlib import Path
from functools import singledispatch
from typing import cast

import rich
import click

from . import ast
from .misc import TypeError
from .click_helpers import simulation_file_option
from .types import FunctionType, is_convertable_to
from .parse_tree import mk_pt
from .ast import mk_ast


@singledispatch
def check(x):
    print(f"Check undefined type={type(x)}")


@check.register
def _(x: ast.Expression):
    # Get everyone's type
    type = x.type

    # If function call we need to do additional checks
    if isinstance(x, ast.FunctionCall):
        type = cast(FunctionType, type)
        if len(x.args) != len(type.params):
            raise TypeError(
                f"Argument count mismatch: n_args={len(x.args)}, n_params={len(type.params)}",
                x.pos,
            )
        for i, (arg, param) in enumerate(zip(x.args, type.params), 1):
            if not is_convertable_to(arg.type, param):
                raise TypeError(
                    f"Type error for argument {i}: arg_type={arg.type.name}, param_type={param.name}"
                )

@check.register
def _(x: ast.Statement):
    match x:
        case ast.PassStatement():
            pass
        case ast.ReturnStatement():
            check(x.expression)
        case ast.IfStatement():
            if 



@check.register
def _(x: ast.GlobalVariable):
    check(x.default)


@check.register
def _(x: ast.Source):
    for g in x.globals:
        check(g)
    for f in x.functions:
        check(f)
    check(x.node_table)
    check(x.edge_table)
    for c in x.contagions:
        check(c)


@click.command()
@simulation_file_option
def print_checked_ast(simulation_file: Path):
    """Print the type checked AST."""
    file_bytes = simulation_file.read_bytes()
    pt = mk_pt(str(simulation_file), file_bytes)
    ast = mk_ast(simulation_file, pt)
    check(ast)
    rich.print(ast)
