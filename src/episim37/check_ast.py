"""Run correctness checks on AST."""

from pathlib import Path
from typing import Any, cast

from typeguard import TypeCheckError, check_type

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
    Expression,
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


def is_scalar_type(r: Reference) -> bool:
    match r.value:
        case BuiltinType() as t:
            if t.name in SCALAR_TYPES:
                return True

    return False


def is_table_type(r: Reference) -> bool:
    match r.value:
        case BuiltinType() as t:
            if t.name in TABLE_TYPES:
                return True

    return False


def is_int_type(r: Reference) -> bool:
    match r.value:
        case BuiltinType() as t:
            if t.name in INT_TYPES:
                return True

    return False


def is_float_type(r: Reference) -> bool:
    match r.value:
        case BuiltinType() as t:
            if t.name in FLOAT_TYPES:
                return True

    return False


def is_bool_type(r: Reference) -> bool:
    match r.value:
        case BuiltinType() as t:
            if t.name == "bool":
                return True

    return False


def is_node_type(r: Reference) -> bool:
    match r.value:
        case BuiltinType() as t:
            if t.name == "node":
                return True

    return False


def is_edge_type(r: Reference) -> bool:
    match r.value:
        case BuiltinType() as t:
            if t.name == "edge":
                return True

    return False


def is_node_set(r: Reference) -> bool:
    match r.value:
        case NodeSet():
            return True
        case BuiltinNodeset():
            return True

    return False


def is_edge_set(r: Reference) -> bool:
    match r.value:
        case EdgeSet():
            return True
        case BuiltinEdgeset():
            return True

    return False


def ensure_node_or_edgeset(r: Reference, allow_builtin: bool = False):
    match r.value:
        case NodeSet() | EdgeSet():
            return
        case BuiltinNodeset() | BuiltinEdgeset():
            if allow_builtin:
                return

    raise EslError(
        "Incompatible value",
        f"Expected a node or edgeset; builtin allowed={allow_builtin!s}.",
        r.pos,
    )


def ensure_table_func(r: Reference):
    match r.value:
        case Function() as f:
            if len(f.params) == 1 and is_table_type(f.params[0].type):
                return

    raise EslError(
        "Incompatible function",
        "Expected function with single parameter of type node or edge.",
        r.pos,
    )


def ensure_node_func(r: Reference):
    match r.value:
        case Function() as f:
            if len(f.params) == 1 and is_node_type(f.params[0].type):
                return

    raise EslError(
        "Incompatible function",
        "Expected function with single parameter of type node.",
        r.pos,
    )


def ensure_edge_func(r: Reference):
    match r.value:
        case Function() as f:
            if len(f.params) == 1 and is_edge_type(f.params[0].type):
                return

    raise EslError(
        "Incompatible function",
        "Expected function with single parameter of type edge.",
        r.pos,
    )


def ensure_no_params(r: Reference):
    match r.value:
        case Function() as f:
            if not f.params:
                return

    raise EslError(
        "Incompatible function", "Expected function with no parameters.", r.pos
    )


def ensure_int_func(r: Reference):
    match r.value:
        case Function() as f:
            if f.rtype is not None and is_int_type(f.rtype):
                return

    raise EslError(
        "Incompatible function",
        "Expected function with return type int.",
        r.pos,
    )


def ensure_float_func(r: Reference):
    match r.value:
        case Function() as f:
            if f.rtype is not None and is_float_type(f.rtype):
                return

    raise EslError(
        "Incompatible function",
        "Expected function with return type float.",
        r.pos,
    )


def ensure_bool_func(r: Reference):
    match r.value:
        case Function() as f:
            if f.rtype is not None and is_bool_type(f.rtype):
                return

    raise EslError(
        "Incompatible function",
        "Expected function with return type bool.",
        r.pos,
    )


def ensure_scalar_func(r: Reference):
    match r.value:
        case Function() as f:
            if f.rtype is not None and is_scalar_type(f.rtype):
                return

    raise EslError(
        "Incompatible function",
        "Expected function with a scalar return type.",
        r.pos,
    )


def ensure_void_func(r: Reference):
    match r.value:
        case Function() as f:
            if f.rtype is None:
                return

    raise EslError(
        "Incompatible function",
        "Expected function with no return type.",
        r.pos,
    )


# FIXME: this is a stub
def ensure_kernel_func(r: Reference):
    match r.value:
        case Function():
            return

    raise EslError(
        "Incompatible function",
        "Expected a valid kernel function",
        r.pos,
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
        raise EslError("Incompatible sets", "Sets must be of same type.", stmt.pos)


def check_apply_statement(stmt: ApplyStatement):
    ensure_node_or_edgeset(stmt.set, allow_builtin=True)
    ensure_table_func(stmt.function)

    if is_node_set(stmt.set):
        ensure_node_func(stmt.function)
    elif is_edge_set(stmt.set):
        ensure_edge_func(stmt.function)

    ensure_void_func(stmt.function)
    ensure_kernel_func(stmt.function)


def check_reduce_outvar(r: Reference):
    match r.value:
        case Param() | Variable() | Global() as t:
            if is_scalar_type(t.type):
                return

    raise EslError(
        "Incompatible variable",
        "Output of reduce statements can only be set to scalar variables.",
        r.pos,
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
    if not f.params and f.rtype == None:
        return

    raise EslError(
        "Bad intervene",
        "Intervene function must have no parameters and not return type",
        f.pos,
    )


def check_susceptibility(o: Expression):
    if is_fn_ref(o):
        oref = cast(Reference, o)
        ensure_node_func(oref)
        ensure_float_func(oref)
        ensure_kernel_func(oref)


def check_infectivity(o: Expression):
    if is_fn_ref(o):
        oref = cast(Reference, o)
        ensure_node_func(oref)
        ensure_float_func(oref)
        ensure_kernel_func(oref)


def check_transmissibility(o: Expression):
    if is_fn_ref(o):
        oref = cast(Reference, o)
        ensure_edge_func(oref)
        ensure_float_func(oref)
        ensure_kernel_func(oref)


def check_enabled(o: Expression):
    if is_fn_ref(o):
        oref = cast(Reference, o)
        ensure_edge_func(oref)
        ensure_bool_func(oref)
        ensure_kernel_func(oref)


def check_pexpr(o: Expression | None):
    if o is None:
        return

    if is_fn_ref(o):
        oref = cast(Reference, o)
        ensure_node_func(oref)
        ensure_float_func(oref)
        ensure_kernel_func(oref)


def check_dwell_expr(o: Expression):
    if is_fn_ref(o):
        oref = cast(Reference, o)
        ensure_node_func(oref)
        ensure_float_func(oref)
        ensure_kernel_func(oref)


def check_function_call(fc: FunctionCall):
    func = fc.function.value
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
    match const_ref.value:
        case EnumConstant() as const:
            if const.type != type:
                raise EslError(
                    "Incompatibale value",
                    f"Expected constant of type {type.name}",
                    const_ref.pos,
                )


def is_fn_ref(o: Any) -> bool:
    match o:
        case Reference() as r:
            match r.value:
                case Function():
                    return True

    return False


def is_dist_ref(o: Any) -> bool:
    match o:
        case Reference() as r:
            try:
                check_type(r.value, Distribution)
                return True
            except TypeCheckError:
                pass

    return False


def check_contagion(c: Contagion):
    state_type = c.state_type.value
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
        check_pexpr(transition.pexpr)
        check_dwell_expr(transition.dwell)

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
