"""Run correctness checks on AST."""

from pathlib import Path
from typing import Any, Callable

from .click_helpers import simulation_file_option
from .misc import EslError, SourcePosition
from .parse_tree import mk_pt, ParseTreeConstructionError
from .ast import (
    INT_TYPES,
    FLOAT_TYPES,
    SCALAR_TYPES,
    TABLE_TYPES,
    BuiltinEdgeset,
    BuiltinFunction,
    BuiltinNodeset,
    BuiltinType,
    Contagion,
    Distribution,
    EdgeSet,
    EnumConstant,
    EnumType,
    Global,
    NodeSet,
    Param,
    Function,
    Variable,
    FunctionCall,
    SelectStatement,
    SampleStatement,
    ApplyStatement,
    ReduceStatement,
    Reference,
    Source,
    mk_ast,
)

import click
import rich


class UnexpectedValue(EslError):
    def __init__(self, unexpected: Any, expected: str, pos: SourcePosition | None):
        super().__init__("Unexpected value", f"{expected}; got {unexpected}", pos)


def resolve_if_ref(wrapped: Callable) -> Callable:
    def wrapper(x, *args, **kwargs):
        try:
            match x:
                case Reference():
                    return wrapped(x.resolve(), *args, **kwargs)
                case _:
                    return wrapped(x, *args, **kwargs)
        except EslError as e:
            if hasattr(x, "pos"):
                e.pos = x.pos
            raise e

    return wrapper


def check_if_fn_or_fnref(wrapped: Callable[[Function], None]) -> Callable[[Any], None]:
    def wrapper(x):
        try:
            match x:
                case Function() as f:
                    return wrapped(f)
                case Reference() as r:
                    match r.resolve():
                        case Function() as f:
                            return wrapped(f)
        except EslError as e:
            if hasattr(x, "pos"):
                e.pos = x.pos
            raise e

    return wrapper


@resolve_if_ref
def is_scalar_type(o) -> bool:
    match o:
        case BuiltinType() as t:
            if t.name in SCALAR_TYPES:
                return True

    return False


@resolve_if_ref
def is_table_type(o) -> bool:
    match o:
        case BuiltinType() as t:
            if t.name in TABLE_TYPES:
                return True

    return False


@resolve_if_ref
def is_int_type(o) -> bool:
    match o:
        case BuiltinType() as t:
            if t.name in INT_TYPES:
                return True

    return False


@resolve_if_ref
def is_float_type(o) -> bool:
    match o:
        case BuiltinType() as t:
            if t.name in FLOAT_TYPES:
                return True

    return False


@resolve_if_ref
def is_bool_type(o) -> bool:
    match o:
        case BuiltinType() as t:
            if t.name == "bool":
                return True

    return False


@resolve_if_ref
def is_node_type(o) -> bool:
    match o:
        case BuiltinType() as t:
            if t.name == "node":
                return True

    return False


@resolve_if_ref
def is_edge_type(o) -> bool:
    match o:
        case BuiltinType() as t:
            if t.name == "edge":
                return True

    return False


@resolve_if_ref
def is_node_set(s) -> bool:
    match s:
        case NodeSet():
            return True
        case BuiltinNodeset():
            return True

    return False


@resolve_if_ref
def is_edge_set(s) -> bool:
    match s:
        case EdgeSet():
            return True
        case BuiltinEdgeset():
            return True

    return False


@resolve_if_ref
def ensure_node_or_edgeset(o, allow_builtin: bool = False):
    match o:
        case NodeSet() | EdgeSet():
            return
        case BuiltinNodeset() | BuiltinEdgeset():
            if allow_builtin:
                return

    raise EslError(
        "Invalid value",
        f"Expected a node or edgeset; builtin allowed {allow_builtin}.",
        None,
    )


@resolve_if_ref
def ensure_table_func(o):
    match o:
        case Function() as f:
            if len(f.params) == 1 and is_table_type(f.params[0].type):
                return

    raise EslError(
        "Incompatible function",
        "Expected function with single parameter of type node or edge.",
        None,
    )


@resolve_if_ref
def ensure_node_func(o):
    match o:
        case Function() as f:
            if len(f.params) == 1 and is_node_type(f.params[0].type):
                return

    raise EslError(
        "Incompatible function",
        "Expected function with single parameter of type node.",
        None,
    )


@resolve_if_ref
def ensure_edge_func(o):
    match o:
        case Function() as f:
            if len(f.params) == 1 and is_edge_type(f.params[0].type):
                return

    raise EslError(
        "Incompatible function",
        "Expected function with single parameter of type edge.",
        None,
    )


@resolve_if_ref
def ensure_no_params(o):
    match o:
        case Function() as f:
            if not f.params:
                return

    raise EslError(
        "Incompatible function",
        "Expected function with no parameters.",
        None,
    )


@resolve_if_ref
def ensure_int_func(o):
    match o:
        case Function() as f:
            if is_int_type(f.rtype):
                return

    raise EslError(
        "Incompatible function",
        "Expected function with return type int.",
        None,
    )


@resolve_if_ref
def ensure_float_func(o):
    match o:
        case Function() as f:
            if is_float_type(f.rtype):
                return

    raise EslError(
        "Incompatible function",
        "Expected function with return type float.",
        None,
    )


@resolve_if_ref
def ensure_bool_func(o):
    match o:
        case Function() as f:
            if is_bool_type(f.rtype):
                return

    raise EslError(
        "Incompatible function",
        "Expected function with return type bool.",
        None,
    )


@resolve_if_ref
def ensure_scalar_func(o):
    match o:
        case Function() as f:
            if is_scalar_type(f.rtype):
                return

    raise EslError(
        "Incompatible function",
        "Expected function with a scalar return type.",
        None,
    )


@resolve_if_ref
def ensure_void_func(o):
    match o:
        case Function() as f:
            if f.rtype is None:
                return

    raise EslError(
        "Incompatible function",
        "Expected function with no return type.",
        None,
    )


# FIXME: this is a stub
@resolve_if_ref
def ensure_kernel_func(o):
    match o:
        case Function():
            return

    raise EslError(
        "Incompatible function",
        "Expected a valid kernel function",
        None,
    )


def check_select_statement(stmt: SelectStatement):
    ensure_node_or_edgeset(stmt.set)
    ensure_table_func(stmt.function)

    if is_node_set(stmt.set):
        ensure_node_func(stmt.function)
    elif is_edge_set(stmt.set):
        ensure_edge_func(stmt.function)

    ensure_bool_func(stmt.function)
    ensure_kernel_func(stmt.function)


def check_sample_statement(stmt: SampleStatement):
    ensure_node_or_edgeset(stmt.set)
    ensure_node_or_edgeset(stmt.parent, allow_builtin=True)

    if is_node_set(stmt.set) and is_node_set(stmt.parent):
        pass
    elif is_edge_set(stmt.set) and is_edge_set(stmt.parent):
        pass
    else:
        raise EslError("Incompatible sets", "Sets must be of same type", stmt.pos)


def check_apply_statement(stmt: ApplyStatement):
    ensure_node_or_edgeset(stmt.set, allow_builtin=True)
    ensure_table_func(stmt.function)

    if is_node_set(stmt.set):
        ensure_node_func(stmt.function)
    elif is_edge_set(stmt.set):
        ensure_edge_func(stmt.function)

    ensure_void_func(stmt.function)
    ensure_kernel_func(stmt.function)


@resolve_if_ref
def check_reduce_outvar(o):
    match o:
        case Param() | Variable() | Global() as t:
            if is_scalar_type(t.type):
                return

    raise EslError(
        "Incompatible var",
        "Output of reduce statements can only be set to scalar variables",
        None,
    )


def check_reduce_statement(stmt: ReduceStatement):
    ensure_node_or_edgeset(stmt.set, allow_builtin=True)
    check_reduce_outvar(stmt.outvar)

    if is_node_set(stmt.set):
        ensure_node_func(stmt.function)
    elif is_edge_set(stmt.set):
        ensure_edge_func(stmt.function)
    else:
        raise UnexpectedValue(stmt.set, "Node or edgeset reference", stmt.set.pos)

    ensure_scalar_func(stmt.function)
    ensure_kernel_func(stmt.function)


def check_intervene(f: Function):
    ensure_no_params(f)
    ensure_void_func(f)


@check_if_fn_or_fnref
def check_susceptibility(f: Function):
    ensure_node_func(f)
    ensure_float_func(f)
    ensure_kernel_func(f)


@check_if_fn_or_fnref
def check_infectivity(f: Function):
    ensure_node_func(f)
    ensure_float_func(f)
    ensure_kernel_func(f)


@check_if_fn_or_fnref
def check_transmissibility(f: Function):
    ensure_edge_func(f)
    ensure_float_func(f)
    ensure_kernel_func(f)


@check_if_fn_or_fnref
def check_enabled(f: Function):
    ensure_edge_func(f)
    ensure_bool_func(f)
    ensure_kernel_func(f)


@check_if_fn_or_fnref
def check_pfunc_or_dwell(f: Function):
    if f.params:
        ensure_node_func(f)
    ensure_float_func(f)
    ensure_kernel_func(f)


def check_function_call(fc: FunctionCall):
    func = fc.function.resolve()
    n_args = len(fc.args)
    match func:
        case Function() | BuiltinFunction() as f:
            n_params = len(f.params)
            if n_params != n_args:
                raise EslError(
                    "Invalid args",
                    f"Expected {n_params} args; got {n_args}.",
                    fc.pos,
                )
        case f if isinstance(f, Distribution):
            if n_args != 0:
                raise EslError(
                    "Invalid args",
                    f"Expected 0 args; got {n_args}.",
                    fc.pos,
                )
        case _ as unexpected:
            raise UnexpectedValue(
                unexpected, "Expected a function or distribution", fc.function.pos
            )


def ensure_enum_compat(type: EnumType, const_ref: Reference):
    match const_ref.resolve():
        case EnumConstant() as const:
            if const.type != type:
                raise EslError(
                    "Incompatibale value",
                    f"Expected constant of type {type.name}",
                    const_ref.pos,
                )


def check_contagion(c: Contagion):
    state_type = c.state_type.resolve()
    match state_type:
        case EnumType():
            pass
        case _:
            raise EslError(
                "Invalid type", "State type needs to be a enum type", c.state_type.pos
            )

    for transition in c.transitions:
        ensure_enum_compat(state_type, transition.entry)
        ensure_enum_compat(state_type, transition.exit)
        check_pfunc_or_dwell(transition.pfunc)
        check_pfunc_or_dwell(transition.dwell)

    for transmission in c.transmissions:
        ensure_enum_compat(state_type, transmission.contact)
        ensure_enum_compat(state_type, transmission.entry)
        ensure_enum_compat(state_type, transmission.exit)

    check_susceptibility(c.susceptibility)
    check_infectivity(c.infectivity)
    check_transmissibility(c.transmissibility)
    check_enabled(c.enabled)


def check_ast(source: Source):
    """Check the generated AST."""

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
    except (ParseTreeConstructionError, EslError) as e:
        e.rich_print()
        raise SystemExit(1)
