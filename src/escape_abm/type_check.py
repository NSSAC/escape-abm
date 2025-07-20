"""Check for types."""

from pathlib import Path
from functools import singledispatch
from typing import assert_never

import rich
import click

from . import ast
from .misc import TypeError
from .click_helpers import simulation_file_option
from .parse_tree import mk_pt
from .ast import mk_ast
from .types import (
    get_type,
    is_convertable_to,
    is_function_type,
    is_numeric_type,
    is_integral_type,
    is_boolean_type,
    is_eq_compareable,
    is_ord_compareable,
    lub_type,
)


@singledispatch
def check_type(x):
    print(f"Check undefined for type={type(x)}")


@check_type.register
def _(x: ast.Expression):
    # If expression has a concrete type
    # we don't need to check its type again
    if x.type.name != "_type":
        pass

    match x:
        case ast.Literal():
            pass
        case ast.Reference():
            deref = x.deref()
            try:
                if isinstance(deref, tuple):
                    type = deref[-1].type
                else:
                    type = deref.type
                x.type = type
            except Exception as e:
                raise TypeError(f"Failed to get type for {x.name}", x.pos) from e

        case ast.UnaryExpression():
            check_type(x.argument)
            type = x.argument.type

            if x.operator == "not":
                if not is_boolean_type(type):
                    raise TypeError(f"Expected boolean type; got {type.name}.")
                x.type = get_type("bool")
            elif x.operator in ["+", "-"]:
                if not is_numeric_type(type):
                    raise TypeError(f"Expected numeric type; got {type.name}.")
                x.type = type
            else:
                raise RuntimeError(f"Unexpected unary operator {x.operator}")

        case ast.BinaryExpression():
            check_type(x.left)
            check_type(x.right)
            ltype = x.left.type
            rtype = x.right.type

            if x.operator in ["+", "-", "*", "/"]:
                if not is_numeric_type(ltype):
                    raise TypeError(
                        f"Expected numeric expression; got {ltype.name}", x.left.pos
                    )
                if not is_numeric_type(rtype):
                    raise TypeError(
                        f"Expected numeric expression; got {rtype.name}", x.right.pos
                    )
                x.type = lub_type(ltype, rtype)
            elif x.operator == "%":
                if not is_integral_type(ltype):
                    raise TypeError(
                        f"Expected integral expression; got {ltype.name}", x.left.pos
                    )
                if not is_integral_type(rtype):
                    raise TypeError(
                        f"Expected integral expression; got {rtype.name}", x.right.pos
                    )
                x.type = lub_type(ltype, rtype)
            elif x.operator in ["and", "or"]:
                if not is_boolean_type(ltype):
                    raise TypeError(
                        f"Expected boolean expression; got {ltype.name}", x.left.pos
                    )
                if not is_boolean_type(rtype):
                    raise TypeError(
                        f"Expected boolean expression; got {rtype.name}", x.right.pos
                    )
                x.type = get_type("bool")
            elif x.operator in ["==", "!="]:
                if not is_eq_compareable(ltype, rtype):
                    raise TypeError(
                        f"Uncompareable types: {ltype.name}, {rtype.name}", x.pos
                    )
                x.type = get_type("bool")
            elif x.operator in ["<", "<=", ">", ">="]:
                if not is_ord_compareable(ltype, rtype):
                    raise TypeError(
                        f"Unorderable types: {ltype.name}, {rtype.name}", x.pos
                    )
                x.type = get_type("bool")
            else:
                raise RuntimeError(f"Unexpected binary operator {x.operator}")

        case ast.ParenthesizedExpression():
            check_type(x.expression)
            x.type = x.expression.type

        case ast.FunctionCall():
            check_type(x.function)
            for arg in x.args:
                check_type(arg)

            fn_type = x.function.type
            if not is_function_type(fn_type):
                raise TypeError(
                    f"Expected function reference: got {fn_type.name}", x.function.pos
                )

            if len(fn_type.params) != len(x.args):
                raise TypeError(
                    f"Expected {len(fn_type.params)} args; got {len(x.args)}", x.pos
                )

            for i, (typ, expr) in enumerate(zip(fn_type.params, x.args), 1):
                if not is_convertable_to(expr.type, typ):
                    raise TypeError(
                        f"Incompatible {i}th argument: got expression of type {expr.type.name}; expected {typ.name}."
                    )

            x.type = fn_type.return_
        case _:
            assert_never("Should be unreachable")


@check_type.register
def _(x: ast.Statement):
    match x:
        case ast.PassStatement():
            pass

        case ast.CallStatement():
            check_type(x.call)

        case ast.ReturnStatement():
            check_type(x.expression)

        case ast.IfStatement():
            check_type(x.condition)
            for s in x.body:
                check_type(s)
            for elif_ in x.elifs:
                check_type(elif_.condition)
                for s in elif_.body:
                    check_type(s)
            if x.else_ is not None:
                for s in x.else_.body:
                    check_type(s)

            if not is_numeric_type(x.condition.type):
                raise TypeError(
                    f"Expected boolean or numeric type; got {x.condition.type.name}",
                    x.condition.pos,
                )
            for elif_ in x.elifs:
                if not is_numeric_type(elif_.condition.type):
                    raise TypeError(
                        f"Expected boolean or numeric type; {elif_.condition.type.name}",
                        elif_.condition.pos,
                    )

        case ast.SwitchStatement():
            check_type(x.condition)
            for case_ in x.cases:
                check_type(case_.match)
                for s in case_.body:
                    check_type(s)
            if x.default is not None:
                for s in x.default.body:
                    check_type(s)

            for case_ in x.cases:
                if not is_eq_compareable(x.condition.type, case_.match.type):
                    raise TypeError(
                        f"Can't compare expression of types {case_.match.type.name} and {x.condition.type.name}",
                        case_.match.pos,
                    )

        case ast.WhileLoop():
            check_type(x.condition)
            for s in x.body:
                check_type(s)

            if not is_numeric_type(x.condition.type):
                raise TypeError(
                    f"Expected boolean or numeric type: got {x.condition.type.name}",
                    x.condition.pos,
                )

        case ast.AssignmentStatement():
            check_type(x.lvalue)
            check_type(x.rvalue)

            if not is_convertable_to(x.rvalue.type, x.lvalue.type):
                raise TypeError(
                    f"Can't assign expression of type {x.lvalue.type.name} to {x.rvalue.type.name}",
                    x.pos,
                )

        case ast.UpdateStatement():
            check_type(x.lvalue)
            check_type(x.rvalue)

            if x.operator in ["+=", "-=", "*=", "/="]:
                if not is_numeric_type(x.lvalue.type):
                    raise TypeError(
                        f"Expected numeric variable; got {x.lvalue.type.name}",
                        x.lvalue.pos,
                    )
                if not is_numeric_type(x.rvalue.type):
                    raise TypeError(
                        f"Expected numeric expression; got {x.rvalue.type.name}",
                        x.rvalue.pos,
                    )

            if x.operator == "%=":
                if not is_integral_type(x.lvalue.type):
                    raise TypeError(
                        f"Expected integral variable; got {x.lvalue.type.name}",
                        x.lvalue.pos,
                    )
                if not is_integral_type(x.rvalue.type):
                    raise TypeError(
                        f"Expected integral expression; got {x.rvalue.type.name}",
                        x.rvalue.pos,
                    )

        case ast.ParallelStatement():
            pass


@check_type.register
def _(x: ast.Source):
    for g in x.globals:
        check_type(g.default)
    for f in x.functions:
        for s in f.body:
            check_type(s)
    for c in x.contagions:
        check_type(c)


@click.command()
@simulation_file_option
def print_checked_ast(simulation_file: Path):
    """Print the type checked AST."""
    file_bytes = simulation_file.read_bytes()
    pt = mk_pt(str(simulation_file), file_bytes)
    ast = mk_ast(simulation_file, pt)
    check_type(ast)
    rich.print(ast)
