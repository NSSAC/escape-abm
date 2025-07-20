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
    Type,
    get_type,
    is_enum_type,
    lub_type,
    is_convertable_to,
    is_function_type,
    is_numeric_type,
    is_integral_type,
    is_boolean_type,
    is_eq_compareable,
    is_ord_compareable,
    is_compatible_function,
    make_fn_type,
)


@singledispatch
def check_type(x) -> None:
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
                    raise TypeError(
                        f"Expected boolean type; got {type}.", x.argument.pos
                    )
                x.type = get_type("bool")
            elif x.operator in ["+", "-"]:
                if not is_numeric_type(type):
                    raise TypeError(
                        f"Expected numeric type; got {type}.", x.argument.pos
                    )
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
                        f"Expected numeric expression; got {ltype}", x.left.pos
                    )
                if not is_numeric_type(rtype):
                    raise TypeError(
                        f"Expected numeric expression; got {rtype}", x.right.pos
                    )
                x.type = lub_type(ltype, rtype)
            elif x.operator == "%":
                if not is_integral_type(ltype):
                    raise TypeError(
                        f"Expected integral expression; got {ltype}", x.left.pos
                    )
                if not is_integral_type(rtype):
                    raise TypeError(
                        f"Expected integral expression; got {rtype}", x.right.pos
                    )
                x.type = lub_type(ltype, rtype)
            elif x.operator in ["and", "or"]:
                if not is_boolean_type(ltype):
                    raise TypeError(
                        f"Expected boolean expression; got {ltype}", x.left.pos
                    )
                if not is_boolean_type(rtype):
                    raise TypeError(
                        f"Expected boolean expression; got {rtype}", x.right.pos
                    )
                x.type = get_type("bool")
            elif x.operator in ["==", "!="]:
                if not is_eq_compareable(ltype, rtype):
                    raise TypeError(f"Uncompareable types: {ltype}, {rtype}", x.pos)
                x.type = get_type("bool")
            elif x.operator in ["<", "<=", ">", ">="]:
                if not is_ord_compareable(ltype, rtype):
                    raise TypeError(f"Unorderable types: {ltype}, {rtype}", x.pos)
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
                    f"Expected function reference: got {fn_type}", x.function.pos
                )

            if len(fn_type.params) != len(x.args):
                raise TypeError(
                    f"Expected {len(fn_type.params)} args; got {len(x.args)}", x.pos
                )

            for i, (ptype, expr) in enumerate(zip(fn_type.params, x.args), 1):
                if not is_convertable_to(expr.type, ptype):
                    raise TypeError(
                        f"Incompatible {i}th argument: got expression of type {expr.type}; expected {ptype}."
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
                    f"Expected boolean or numeric type; got {x.condition.type}",
                    x.condition.pos,
                )
            for elif_ in x.elifs:
                if not is_numeric_type(elif_.condition.type):
                    raise TypeError(
                        f"Expected boolean or numeric type; {elif_.condition.type}",
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
                        f"Can't compare expression of types {case_.match.type} and {x.condition.type}",
                        case_.match.pos,
                    )

        case ast.WhileLoop():
            check_type(x.condition)
            for s in x.body:
                check_type(s)

            if not is_numeric_type(x.condition.type):
                raise TypeError(
                    f"Expected boolean or numeric type: got {x.condition.type}",
                    x.condition.pos,
                )

        case ast.AssignmentStatement():
            check_type(x.lvalue)
            check_type(x.rvalue)

            if not is_convertable_to(x.rvalue.type, x.lvalue.type):
                raise TypeError(
                    f"Can't assign expression of type {x.lvalue.type} to {x.rvalue.type}",
                    x.pos,
                )

        case ast.UpdateStatement():
            check_type(x.lvalue)
            check_type(x.rvalue)

            if x.operator in ["+=", "-=", "*=", "/="]:
                if not is_numeric_type(x.lvalue.type):
                    raise TypeError(
                        f"Expected numeric variable; got {x.lvalue.type}",
                        x.lvalue.pos,
                    )
                if not is_numeric_type(x.rvalue.type):
                    raise TypeError(
                        f"Expected numeric expression; got {x.rvalue.type}",
                        x.rvalue.pos,
                    )

            if x.operator == "%=":
                if not is_integral_type(x.lvalue.type):
                    raise TypeError(
                        f"Expected integral variable; got {x.lvalue.type}",
                        x.lvalue.pos,
                    )
                if not is_integral_type(x.rvalue.type):
                    raise TypeError(
                        f"Expected integral expression; got {x.rvalue.type}",
                        x.rvalue.pos,
                    )

        case ast.ParallelStatement():
            if x.filter_clause is not None:
                fn = x.filter_clause.function
                check_type(fn)

                if is_function_type(fn.type):
                    expected_type = make_fn_type([x.table], "float")
                    if not is_compatible_function(fn.type, expected_type):
                        raise TypeError(
                            f"Incompatible function; expected {expected_type} got {fn.type}",
                            fn.pos,
                        )
                else:
                    if not is_convertable_to(fn.type, get_type("float")):
                        raise TypeError(
                            f"Incompatible expression type; expected float got {fn.type}",
                            fn.pos,
                        )
            if x.sample_clause is not None:
                check_type(x.sample_clause.amount)

                if not is_numeric_type(x.sample_clause.amount.type):
                    raise TypeError(
                        f"Expected numeric expression; got {x.sample_clause.amount.type}",
                        x.sample_clause.amount.pos,
                    )

            if x.apply_clause is not None:
                fn = x.apply_clause.function
                check_type(fn)

                if is_function_type(fn.type):
                    expected_type = make_fn_type([x.table], "_type")
                    if not is_compatible_function(fn.type, expected_type):
                        raise TypeError(
                            f"Incompatible function; expected node -> void got {fn.type}",
                            fn.pos,
                        )
                else:
                    raise TypeError(
                        f"Apply clause must be provided with a function expression",
                        fn.pos,
                    )

            for clause in x.reduce_clauses:
                check_type(clause.lvalue)
                check_type(clause.function)

                if clause.operator in ["+", "*"]:
                    if not is_numeric_type(clause.lvalue.type):
                        raise TypeError(
                            f"Expected numeric variable; got {clause.lvalue.type}",
                            clause.lvalue.pos,
                        )
                    if is_function_type(clause.function.type):
                        if not is_numeric_type(clause.function.type.return_):
                            raise TypeError(
                                f"Expected numeric function; got {clause.function.type}",
                                clause.function.pos,
                            )
                    else:
                        if not is_numeric_type(clause.function.type):
                            raise TypeError(
                                f"Expected numeric expression; got {clause.function.type}",
                                clause.function.pos,
                            )


@check_type.register
def _(x: ast.Function | ast.LambdaFunction):
    assert is_function_type(x.type)

    for s in x.body:
        check_type(s)

    for s in x.return_statements:
        if not is_convertable_to(s.expression.type, x.type.return_):
            raise TypeError(
                f"Invalid return type {s.expression.type} for function of type {x.type}",
                s.expression.pos,
            )

    if x.type.return_ != "void" and not x.return_statements:
        raise TypeError(f"Non void function with not return statements", x.pos)


def _check_enum_constant(x: ast.Reference, type: Type):
    if not isinstance(x.ref, ast.EnumConstant) or x.type != x.type:
        raise TypeError(
            f"Expected enumeration constant of type {type}; got {x.ref}",
            x.pos,
        )


@check_type.register
def _(x: ast.Contagion):
    for t in x.transitions:
        check_type(t.entry)
        check_type(t.exit)
    for t in x.transmissions:
        check_type(t.contact)
        check_type(t.entry)
        check_type(t.exit)
    check_type(x.transition_rate)
    check_type(x.dwell_time)
    check_type(x.susceptibility)
    check_type(x.infectivity)
    check_type(x.transmissibility)

    if not is_enum_type(x.state_type):
        raise TypeError(f"Expected an type got {x.state_type}", x.pos)

    for t in x.transitions:
        _check_enum_constant(t.entry, x.state_type)
        _check_enum_constant(t.exit, x.state_type)
    for t in x.transmissions:
        _check_enum_constant(t.contact, x.state_type)
        _check_enum_constant(t.entry, x.state_type)
        _check_enum_constant(t.exit, x.state_type)

    for fn in [x.transition_rate, x.dwell_time]:
        if is_function_type(fn.type):
            expected_type = make_fn_type(["node", x.state_type.name], "float")
            if not is_compatible_function(fn.type, expected_type):
                raise TypeError(
                    f"Incompatible function; expected {expected_type} got {fn.type}",
                    fn.pos,
                )
        else:
            if not is_convertable_to(fn.type, get_type("float")):
                raise TypeError(
                    f"Incompatible expression type; expected float got {fn.type}",
                    fn.pos,
                )

    for fn in [x.susceptibility, x.infectivity]:
        if is_function_type(fn.type):
            expected_type = make_fn_type(["node"], "float")
            if not is_compatible_function(fn.type, expected_type):
                raise TypeError(
                    f"Incompatible function; expected {expected_type} got {fn.type}",
                    fn.pos,
                )
        else:
            if not is_convertable_to(fn.type, get_type("float")):
                raise TypeError(
                    f"Incompatible expression type; expected float got {fn.type}",
                    fn.pos,
                )

    fn = x.transmissibility
    if is_function_type(fn.type):
        expected_type = make_fn_type(["edge"], "float")
        if not is_compatible_function(fn.type, expected_type):
            raise TypeError(
                f"Incompatible function; expected {expected_type} got {fn.type}",
                fn.pos,
            )
    else:
        if not is_convertable_to(fn.type, get_type("float")):
            raise TypeError(
                f"Incompatible expression type; expected float got {fn.type}",
                fn.pos,
            )


@check_type.register
def _(x: ast.Source):
    for g in x.globals:
        check_type(g.default)

        if not is_convertable_to(g.default.type, g.type):
            raise TypeError(
                f"Invalid default expression of type {g.default.type} for variable {g.type}",
                g.default.pos,
            )
    for f in x.functions:
        check_type(f)
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
