"""Run correctness checks on AST."""

from pathlib import Path
from typing import cast
from contextlib import contextmanager

from typeguard import TypeCheckError, check_type

from .click_helpers import simulation_file_option
from .misc import SourcePosition, EslError, RichException
from .parse_tree import mk_pt
from . import ast
from .ast import mk_ast

import click
import rich


@contextmanager
def err_desc(description: str, pos: SourcePosition | None):
    try:
        yield
    except AssertionError:
        raise EslError("Semantic error", description, pos)


def is_scalar_type(r: ast.Reference) -> bool:
    match r.value:
        case ast.BuiltinType() as t:
            if t.name in ast.SCALAR_TYPES:
                return True

    return False


def is_int_type(r: ast.Reference) -> bool:
    match r.value:
        case ast.BuiltinType() as t:
            if t.name in ast.INT_TYPES:
                return True
        case ast.EnumType():
            return True

    return False


def is_float_type(r: ast.Reference) -> bool:
    match r.value:
        case ast.BuiltinType() as t:
            if t.name in ast.FLOAT_TYPES:
                return True

    return False


def is_bool_type(r: ast.Reference) -> bool:
    match r.value:
        case ast.BuiltinType() as t:
            if t.name == "bool":
                return True

    return False


def is_node_type(r: ast.Reference) -> bool:
    match r.value:
        case ast.BuiltinType() as t:
            if t.name == "node":
                return True

    return False


def is_edge_type(r: ast.Reference) -> bool:
    match r.value:
        case ast.BuiltinType() as t:
            if t.name == "edge":
                return True

    return False


def is_node_set(r: ast.Reference) -> bool:
    match r.value:
        case ast.NodeSet():
            return True
        case ast.BuiltinNodeset():
            return True

    return False


def is_edge_set(r: ast.Reference) -> bool:
    match r.value:
        case ast.EdgeSet():
            return True
        case ast.BuiltinEdgeset():
            return True

    return False


def is_set(r: ast.Reference, allow_builtin: bool = False) -> bool:
    match r.value:
        case ast.NodeSet() | ast.EdgeSet():
            return True
        case ast.BuiltinNodeset() | ast.BuiltinEdgeset():
            if allow_builtin:
                return True

    return False


def is_node_func(r: ast.Reference) -> bool:
    match r.value:
        case ast.Function() as f:
            if len(f.params) == 1 and is_node_type(f.params[0].type):
                return True

    return False


def is_edge_func(r: ast.Reference) -> bool:
    match r.value:
        case ast.Function() as f:
            if len(f.params) == 1 and is_edge_type(f.params[0].type):
                return True

    return False


def is_int_func(r: ast.Reference) -> bool:
    match r.value:
        case ast.Function() as f:
            if f.rtype is not None and is_int_type(f.rtype):
                return True

    return False


def is_float_func(r: ast.Reference) -> bool:
    match r.value:
        case ast.Function() as f:
            if f.rtype is not None and is_float_type(f.rtype):
                return True

    return False


def is_bool_func(r: ast.Reference) -> bool:
    match r.value:
        case ast.Function() as f:
            if f.rtype is not None and is_bool_type(f.rtype):
                return True

    return False


def is_scalar_func(r: ast.Reference) -> bool:
    match r.value:
        case ast.Function() as f:
            if f.rtype is not None and is_scalar_type(f.rtype):
                return True

    return False


# FIXME: this is a stub
def is_kernel_func(r: ast.Reference) -> bool:
    match r.value:
        case ast.Function():
            return True

    return False


def check_select_statement(stmt: ast.SelectStatement):
    with err_desc("Expected reference to a user defined set.", stmt.set.pos):
        assert is_set(stmt.set, allow_builtin=False)

    if is_node_set(stmt.set):
        with err_desc("Expected a node function.", stmt.function.pos):
            assert is_node_func(stmt.function)
    else:  # is_edge_set(stmt.set)
        with err_desc("Expected a edge function.", stmt.function.pos):
            assert is_edge_func(stmt.function)

    with err_desc("Expected a function returning bool.", stmt.function.pos):
        assert is_bool_func(stmt.function)
    with err_desc("Expected a kernel function.", stmt.function.pos):
        assert is_kernel_func(stmt.function)


def check_sample_statement(stmt: ast.SampleStatement):
    with err_desc("Expected reference to a user defined set.", stmt.set.pos):
        assert is_set(stmt.set, allow_builtin=False)
    with err_desc("Expected reference to a set.", stmt.set.pos):
        assert is_set(stmt.parent, allow_builtin=True)

    if is_node_set(stmt.set):
        with err_desc(
            "Node sets can only be sampled from other node sets.", stmt.set.pos
        ):
            assert is_node_set(stmt.parent)
    else:  # is_edge_set(stmt.set)
        with err_desc(
            "Edge sets can only be sampled from other edge sets.", stmt.set.pos
        ):
            assert is_node_set(stmt.parent)


def check_apply_statement(stmt: ast.ApplyStatement):
    with err_desc("Expected reference to a set.", stmt.set.pos):
        assert is_set(stmt.set, allow_builtin=True)

    if is_node_set(stmt.set):
        with err_desc("Expected a node function.", stmt.function.pos):
            assert is_node_func(stmt.function)
    else:  # is_edge_set(stmt.set)
        with err_desc("Expected a edge function.", stmt.function.pos):
            assert is_edge_func(stmt.function)

    with err_desc("Expected a function with no return type.", stmt.function.pos):
        assert stmt.function.value.rtype is None
    with err_desc("Expected a kernel function.", stmt.function.pos):
        assert is_kernel_func(stmt.function)


def is_scalar_var(r: ast.Reference) -> bool:
    match r.value:
        case ast.Param() | ast.Variable() | ast.Global() as t:
            if is_scalar_type(t.type):
                return True

    return False


def check_reduce_statement(stmt: ast.ReduceStatement):
    with err_desc("Expected reference to a set.", stmt.set.pos):
        assert is_set(stmt.set, allow_builtin=True)
    with err_desc("Expect a scalar variable.", stmt.outvar.pos):
        assert is_scalar_var(stmt.outvar)

    if is_node_set(stmt.set):
        with err_desc("Expected a node function.", stmt.function.pos):
            assert is_node_func(stmt.function)
    else:  # is_edge_set(stmt.set)
        with err_desc("Expected a edge function.", stmt.function.pos):
            assert is_edge_func(stmt.function)

    with err_desc("Expected a function with scalar return type.", stmt.function.pos):
        assert is_scalar_func(stmt.function)
    with err_desc("Expected a kernel function.", stmt.function.pos):
        assert is_kernel_func(stmt.function)


def check_intervene(f: ast.Function):
    with err_desc("Intervene function must not take any parameters.", f.pos):
        assert not f.params

    with err_desc("Intervene function must not have a return type.", f.pos):
        assert f.rtype is None


def check_susceptibility(o: ast.Expression):
    if is_fn_ref(o):
        fn = cast(ast.Reference, o)
        with err_desc("Expected a node function.", fn.pos):
            assert is_node_func(fn)
        with err_desc("Expected a function with float return type.", fn.pos):
            assert is_float_func(fn)
        with err_desc("Expected a kernel function.", fn.pos):
            assert is_kernel_func(fn)


check_infectivity = check_susceptibility


def check_transmissibility(o: ast.Expression):
    if is_fn_ref(o):
        fn = cast(ast.Reference, o)
        with err_desc("Expected a edge function.", fn.pos):
            assert is_edge_func(fn)
        with err_desc("Expected a function with float return type.", fn.pos):
            assert is_float_func(fn)
        with err_desc("Expected a kernel function.", fn.pos):
            assert is_kernel_func(fn)


def check_enabled(o: ast.Expression):
    if is_fn_ref(o):
        fn = cast(ast.Reference, o)
        with err_desc("Expected a edge function.", fn.pos):
            assert is_edge_func(fn)
        with err_desc("Expected a function with bool return type.", fn.pos):
            assert is_bool_func(fn)
        with err_desc("Expected a kernel function.", fn.pos):
            assert is_kernel_func(fn)


def check_pexpr(o: ast.Expression):
    if is_fn_ref(o):
        fn = cast(ast.Reference, o)
        with err_desc("Expected a node function.", fn.pos):
            assert is_node_func(fn)
        with err_desc("Expected a function with float return type.", fn.pos):
            assert is_float_func(fn)
        with err_desc("Expected a kernel function.", fn.pos):
            assert is_kernel_func(fn)


check_dwell = check_pexpr


def check_function_call(fc: ast.FunctionCall):
    func = fc.function.value
    n_args = len(fc.args)
    match func:
        case ast.Function() | ast.BuiltinFunction() as f:
            n_params = len(f.params)
            with err_desc(f"Expected {n_params} args; got {n_args}.", fc.pos):
                assert n_params == n_args
        case f if isinstance(f, ast.Distribution):
            with err_desc(f"Expected 0 args; got {n_args}.", fc.pos):
                assert n_args == 0
        case _:
            raise EslError(
                "Unexpected value",
                f"Expected a function or distribution; got {func}",
                fc.function.pos,
            )


def is_enum_compat(type: ast.EnumType, const_ref: ast.Reference) -> bool:
    match const_ref.value:
        case ast.EnumConstant() as const:
            if const.type == type:
                return True

    return False


def is_fn_ref(o: ast.Expression) -> bool:
    match o:
        case ast.Reference() as r:
            match r.value:
                case ast.Function():
                    return True

    return False


def is_dist_ref(o: ast.Expression) -> bool:
    match o:
        case ast.Reference() as r:
            try:
                check_type(r.value, ast.Distribution)
                return True
            except TypeCheckError:
                pass

    return False


def check_contagion(c: ast.Contagion):
    state_type = c.state_type.value
    match state_type:
        case ast.EnumType():
            pass
        case _:
            raise EslError(
                "Invalid type", "State type needs to be a enum type", c.state_type.pos
            )

    for t in c.transitions:
        with err_desc("Incompatible enumeration constant", t.entry.pos):
            assert is_enum_compat(state_type, t.entry)
        with err_desc("Incompatible enumeration constant", t.exit.pos):
            assert is_enum_compat(state_type, t.exit)

        check_pexpr(t.pexpr)
        check_dwell(t.dwell)

    for t in c.transmissions:
        with err_desc("Incompatible enumeration constant", t.contact.pos):
            assert is_enum_compat(state_type, t.contact)
        with err_desc("Incompatible enumeration constant", t.entry.pos):
            assert is_enum_compat(state_type, t.entry)
        with err_desc("Incompatible enumeration constant", t.exit.pos):
            assert is_enum_compat(state_type, t.exit)

    check_susceptibility(c.susceptibility)
    check_infectivity(c.infectivity)
    check_transmissibility(c.transmissibility)
    check_enabled(c.enabled)


def check_ast(source: ast.Source):
    """Check the generated AST."""
    try:
        for c in source.contagions:
            check_contagion(c)

        for fc in source.function_calls:
            check_function_call(fc)

        for stmt in source.select_statements:
            check_select_statement(stmt)

        for stmt in source.sample_statements:
            check_sample_statement(stmt)

        for stmt in source.apply_statements:
            check_apply_statement(stmt)

        for stmt in source.reduce_statements:
            check_reduce_statement(stmt)

        check_intervene(source.intervene)
    except EslError:
        raise
    except Exception as e:
        estr = f"{e.__class__.__name__}: {e!s}"
        raise EslError(
            "AST Check Error", f"Failed to check source: ({estr})", source.pos
        )


@click.command()
@simulation_file_option
def print_checked_ast(simulation_file: Path):
    """Print the checked AST."""
    file_bytes = simulation_file.read_bytes()
    try:
        pt = mk_pt(str(simulation_file), file_bytes)
        ast = mk_ast(simulation_file, pt)
        check_ast(ast)
        rich.print(ast)
    except RichException as e:
        e.rich_print()
        raise SystemExit(1)
