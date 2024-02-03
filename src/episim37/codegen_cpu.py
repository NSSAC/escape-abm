"""Codegen for CPU using OpenMP."""

from __future__ import annotations

import sys
import shlex
from pathlib import Path
from textwrap import dedent
from typing import assert_never
from subprocess import run as do_run
from tempfile import TemporaryDirectory
from collections import defaultdict
from itertools import chain
from importlib.resources import files

import attrs
import click
import jinja2

import rich
import rich.markup
from rich.syntax import Syntax

from .parse_tree import mk_pt, ParseTreeConstructionError, SourcePosition
from .ast1 import mk_ast1, ASTConstructionError
from .alias_table import AliasTable
from . import ast1

INC_CSR_INDPTR_DATASET_NAME = "/incoming/incidence/csr/indptr"
TARGET_NODE_INDEX_DATASET_NAME = "/edge/_target_node_index"
SOURCE_NODE_INDEX_DATASET_NAME = "/edge/_source_node_index"

DEFERRED_TYPES = {
    "int_type": "int64_t",
    "uint_type": "uint64_t",
    "float_type": "float",
    "bool_type": "uint8_t",
    "size_type": "uint64_t",
    "node_index_type": "uint32_t",
    "edge_index_type": "uint64_t",
}


@attrs.define
class DeferredType:
    name: str
    base_ctype: str
    h5_type: str
    base_h5_type: str


ENVIRONMENT = jinja2.Environment(
    loader=jinja2.PackageLoader(package_name="episim37", package_path="templates"),
    undefined=jinja2.StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
)

STATIC_DIR = files("episim37.static")


class CodegenError(Exception):
    def __init__(self, type: str, explanation: str, pos: SourcePosition | None):
        super().__init__()
        self.type = type
        self.explanation = explanation
        self.pos = pos

    def __str__(self):
        return "Failed to construct IR"

    def rich_print(self):
        rich.print(f"[red]{self}[/red]")

        if self.pos is None:
            print(self.explanation)
            return

        etype = f"[red]{self.type}[/red]"
        fpath = f"[yellow]{self.pos.source}[/yellow]"
        line = self.pos.line
        col = self.pos.col
        expl = rich.markup.escape(self.explanation)

        rich.print(f"{etype}:{fpath}:{line}:{col}:{expl}")
        print(self.pos.text)


Error = CodegenError


class UnexpectedValue(CodegenError):
    def __init__(self, unexpected, pos: SourcePosition | None):
        super().__init__("Unexpected value", repr(unexpected), pos)


def smallest_uint_type(max_val: int) -> str | None:
    if max_val < 2**8:
        return "uint8_t"
    elif max_val < 2**16:
        return "uint64_t"
    elif max_val < 2**32:
        return "uint32_t"
    elif max_val < 2**64:
        return "uint64_t"
    else:
        return None


def enum_base_type(t: ast1.EnumType) -> str:
    type = smallest_uint_type(len(t.consts) - 1)
    if type is None:
        raise Error("Large enumeration", "Enum too large", t.pos)
    return type


def ctype(t: str) -> str:
    """ESL type to C type."""

    # fmt: off
    type_map = {
        "int":   "int_type",
        "uint":  "uint_type",
        "float": "float_type",
        "bool":  "bool_type",
        "size":  "size_type",

        "u8":  "uint8_t",
        "u16": "uint16_t",
        "u32": "uint32_t",
        "u64": "uint64_t",

        "i8":  "int8_t",
        "i16": "int16_t",
        "i32": "int32_t",
        "i64": "int64_t",

        "f32": "float",
        "f64": "double",

        "node":  "node_index_type",
        "edge":  "edge_index_type",

        "nodeset":  "nodeset*",
        "edgeset":  "edgeset*",
    }
    # fmt: on

    return type_map[t]


def hdf5_type(t: str) -> str:
    """C type to HDF5 type."""

    # fmt: off
    type_map = {
        "uint8_t": "H5::PredType::NATIVE_UINT8",
        "uint16_t": "H5::PredType::NATIVE_UINT16",
        "uint32_t": "H5::PredType::NATIVE_UINT32",
        "uint64_t": "H5::PredType::NATIVE_UINT64",

        "int8_t": "H5::PredType::NATIVE_INT8",
        "int16_t": "H5::PredType::NATIVE_INT16",
        "int32_t": "H5::PredType::NATIVE_INT32",
        "int64_t": "H5::PredType::NATIVE_INT64",

        "float": "H5::PredType::NATIVE_FLOAT",
        "double": "H5::PredType::NATIVE_DOUBLE",
    }
    # fmt: on

    if t in type_map:
        return type_map[t]

    return t.upper() + "_H5_TYPE"


def mangle(*args: str) -> str:
    if len(args) == 1:
        return "_" + args[0]

    return "_" + "__".join(args)


def cstr_to_ctype_fn(t: str) -> str:
    match t:
        case ("int_type" | "int8_t" | "int16_t" | "int32_t" | "int64_t"):
            return "std::stol"
        case (
            "uint_type"
            | "bool_type"
            | "size_type"
            | "node_index_type"
            | "edge_index_type"
            | "uint8_t"
            | "uint16_t"
            | "uint32_t"
            | "uint64_t"
        ):
            return "std::stoul"
        case ("float_type" | "float" | "double"):
            return "std::stod"
        case _ as unexpected:
            raise ValueError(unexpected)


def typename(t: ast1.EnumType | ast1.BuiltinType) -> str:
    match t:
        case ast1.BuiltinType():
            return ctype(t.name)
        case ast1.EnumType():
            return mangle(t.name)
        case _ as unexpected:
            assert_never(unexpected)


def h5_typename(t: ast1.EnumType | ast1.BuiltinType) -> str:
    match t:
        case ast1.BuiltinType():
            return hdf5_type(ctype(t.name))
        case ast1.EnumType():
            return hdf5_type(enum_base_type(t))
        case _ as unexpected:
            assert_never(unexpected)


def update_globals() -> dict:
    deferred_types = []
    for type, base_ctype in DEFERRED_TYPES.items():
        deferred_types.append(
            DeferredType(type, base_ctype, hdf5_type(type), hdf5_type(base_ctype))
        )

    globals = dict(
        INC_CSR_INDPTR_DATASET_NAME=INC_CSR_INDPTR_DATASET_NAME,
        TARGET_NODE_INDEX_DATASET_NAME=TARGET_NODE_INDEX_DATASET_NAME,
        SOURCE_NODE_INDEX_DATASET_NAME=SOURCE_NODE_INDEX_DATASET_NAME,
        DEFERRED_TYPES=deferred_types,
    )

    return globals


ENVIRONMENT.globals.update(update_globals())


def ref_str(x: ast1.Referable) -> str:
    match x:
        case ast1.EnumConstant():
            return mangle(x.name)
        case ast1.Config():
            return mangle(x.name)
        case ast1.BuiltinConfig():
            return x.name.upper()
        case ast1.Global():
            return mangle(x.name)
        case ast1.BuiltinGlobal():
            return x.name.upper()
        case ast1.Param():
            return mangle(x.name)
        case ast1.Variable():
            return mangle(x.name)
        case ast1.NodeSet():
            return mangle(x.name)
        case ast1.EdgeSet():
            return mangle(x.name)
        case [ast1.Param() | ast1.Variable() as v, ast1.NodeField() as f]:
            v_ref = ref_str(v)
            f_ref = mangle(f.name)
            return f"NODE_TABLE->{f_ref}[{v_ref}]"
        case [ast1.Param() | ast1.Variable() as e, ast1.EdgeField() as f]:
            e_ref = ref_str(e)
            f_ref = mangle(f.name)
            return f"EDGE_TABLE->{f_ref}[{e_ref}]"
        case [
            ast1.Param() | ast1.Variable() as v,
            ast1.Contagion() as c,
            ast1.StateAccessor(),
        ]:
            v_ref = ref_str(v)
            f_ref = mangle(c.name) + "_state"
            return f"NODE_TABLE->{f_ref}[{v_ref}]"
        case [
            ast1.Param() | ast1.Variable() as e,
            ast1.SourceNodeAccessor(),
        ]:
            e_ref = ref_str(e)
            return f"EDGE_TABLE->source_node_index[{e_ref}]"
        case [
            ast1.Param() | ast1.Variable() as e,
            ast1.TargetNodeAccessor(),
        ]:
            e_ref = ref_str(e)
            return f"EDGE_TABLE->target_node_index[{e_ref}]"
        case [
            ast1.Param() | ast1.Variable() as e,
            ast1.SourceNodeAccessor(),
            ast1.NodeField() as f,
        ]:
            e_ref = ref_str(e)
            f_ref = mangle(f.name)
            return f"NODE_TABLE->{f_ref}[EDGE_TABLE->source_node_index[{e_ref}]]"
        case [
            ast1.Param() | ast1.Variable() as e,
            ast1.TargetNodeAccessor(),
            ast1.NodeField() as f,
        ]:
            e_ref = ref_str(e)
            f_ref = mangle(f.name)
            return f"NODE_TABLE->{f_ref}[EDGE_TABLE->target_node_index[{e_ref}]]"
        case _ as unexpected:
            assert_never(unexpected)


def call_str(x: ast1.Callable) -> str:
    match x:
        case ast1.BuiltinFunction():
            return mangle(x.name)
        case ast1.Function():
            return mangle(x.name)
        case _ as unexpected:
            assert_never(unexpected)


def upd_str(left: ast1.Updateable, op: str, right_str: str) -> str:
    l_str = ref_str(left)
    return f"{l_str} {op} {right_str}"


def cpp_operator(t: str) -> str:
    match t:
        case "or":
            return "||"
        case "and":
            return "&&"
        case "not":
            return "!"
        case _:
            return t


def expression_str(e: ast1.Expression) -> str:
    match e:
        case bool():
            return str(int(e))
        case int():
            return str(e)
        case float():
            return str(e)
        case ast1.UnaryExpression():
            op = cpp_operator(e.operator)
            eo = expression_str(e.argument)
            return f"{op}{eo}"
        case ast1.BinaryExpression():
            left = expression_str(e.left)
            op = cpp_operator(e.operator)
            right = expression_str(e.right)
            return f"{left} {op} {right}"
        case ast1.ParenthesizedExpression():
            eo = expression_str(e.expression)
            return f"({eo})"
        case ast1.Reference():
            return ref_str(e.resolve_referable())
        case ast1.FunctionCall():
            function = call_str(e.function.resolve_callable())
            args = [expression_str(a) for a in e.args]
            args = ", ".join(args)
            return f"{function}({args})"
        case _ as unexpected:
            assert_never(unexpected)


@attrs.define
class IndentedLines:
    cur_indent: int
    indents: list[int] = attrs.field(factory=list)
    lines: list[str] = attrs.field(factory=list)

    def append(self, line: str):
        self.indents.append(self.cur_indent)
        self.lines.append(line)

    def extend(self, other: IndentedLines):
        self.indents.extend(other.indents)
        self.lines.extend(other.lines)

    def to_string(self, shift: int = 4) -> str:
        out = []
        for indent, line in zip(self.indents, self.lines):
            indent = indent * shift
            if line:
                out.append(" " * indent)
                out.append(line)
                out.append("\n")
        return "".join(out)


def make_line(pos: SourcePosition | None) -> str:
    if pos is None:
        return ""

    return f'#line {pos.line} "{pos.source}"'


def statement_lines(s: ast1.StatementParts, indent: int) -> IndentedLines:
    lines = IndentedLines(indent)

    match s:
        case ast1.PassStatement():
            lines.append(make_line(s.pos))
            lines.append("/* pass */")
        case ast1.ReturnStatement():
            lines.append(make_line(s.pos))
            expr = expression_str(s.expression)
            lines.append(f"return {expr};")
        case ast1.ElifSection():
            cond = expression_str(s.condition)
            cond_line = "else if (%s) {" % cond
            lines.append(cond_line)
            for body in s.body:
                lines.extend(statement_lines(body, indent + 1))
            lines.append("}")
        case ast1.ElseSection():
            lines.append("else {")
            for body in s.body:
                lines.extend(statement_lines(body, indent + 1))
            lines.append("}")
        case ast1.IfStatement():
            lines.append(make_line(s.pos))
            cond = expression_str(s.condition)
            cond_line = "if (%s) {" % cond
            lines.append(cond_line)
            for body in s.body:
                lines.extend(statement_lines(body, indent + 1))
            lines.append("}")
            for elif_ in s.elifs:
                lines.extend(statement_lines(elif_, indent))
            if s.else_ is not None:
                lines.extend(statement_lines(s.else_, indent))
        case ast1.WhileLoop():
            lines.append(make_line(s.pos))
            cond = expression_str(s.condition)
            cond_line = "while (%s) {" % cond
            lines.append(cond_line)
            for body in s.body:
                lines.extend(statement_lines(body, indent + 1))
            lines.append("}")
        case ast1.CallStatement():
            lines.append(make_line(s.pos))
            call = expression_str(s.call)
            call_line = "%s;" % call
            lines.append(call_line)
        case ast1.UpdateStatement():
            lines.append(make_line(s.pos))
            left = s.left.resolve_updateable()
            op = s.operator
            right = expression_str(s.right)
            lines.append(upd_str(left, op, right) + ";")
        case ast1.PrintStatement():
            args = []
            for arg in s.args:
                if isinstance(arg, str):
                    args.append(arg)
                else:
                    arg = expression_str(arg)
                    arg = f"std::to_string({arg})"
                    args.append(arg)
            args = f" << {s.sep} << ".join(args)
            line = "std::cout << " + args + f" << {s.end} << std::flush;"
            lines.append(line)
        case ast1.Variable():
            lines.append(make_line(s.pos))
            left = ref_str(s)
            right = expression_str(s.init)
            lines.append(f"{left} = {right};")
        case ast1.SelectUsing():
            lines.append(make_line(s.pos))
            lines.append(f"{s.name}();")
        case ast1.SelectApprox():
            lines.append(make_line(s.pos))
            lines.append(f"{s.name}();")
        case ast1.SelectRelative():
            lines.append(make_line(s.pos))
            lines.append(f"{s.name}();")
        case ast1.ForeachStatement():
            lines.append(make_line(s.pos))
            lines.append(f"{s.name}();")
        case _ as unexpected:
            assert_never(unexpected)

    return lines


@attrs.define
class EnumType:
    name: str
    base_type: str
    consts: list[str]
    print_consts: list[str]

    @classmethod
    def make(cls, et: ast1.EnumType) -> EnumType:
        name = typename(et)
        base_type = enum_base_type(et)
        consts = [mangle(c) for c in et.consts]
        return cls(name, base_type, consts, et.consts)


@attrs.define
class Config:
    name: str
    type: str
    from_str_fn: str
    env_var: str
    print_name: str
    default: str

    @classmethod
    def make(cls, c: ast1.Config) -> Config:
        name = ref_str(c)
        type = typename(c.type.resolve())
        from_str_fn = cstr_to_ctype_fn(type)
        env_var = c.name.upper()
        default = expression_str(c.default)
        return cls(name, type, from_str_fn, env_var, c.name, default)


@attrs.define
class Global:
    name: str
    type: str
    default: str

    @classmethod
    def make(cls, c: ast1.Global) -> Global:
        name = ref_str(c)
        type = typename(c.type.resolve())
        default = expression_str(c.default)
        return cls(name, type, default)


@attrs.define
class Field:
    name: str
    print_name: str
    dataset_name: str
    type: str
    h5_type: str
    is_static: bool
    is_used: bool


@attrs.define
class NodeTable:
    fields: list[Field]
    contagions: list[tuple[str, str]]  # contagion name, state_type
    key: str

    @classmethod
    def make(cls, tab: ast1.NodeTable) -> NodeTable:
        fields = []
        for field in tab.fields:
            name = mangle(field.name)
            print_name = field.name
            dataset_name = f"/node/{field.name}"
            type = typename(field.type.resolve())
            h5_type = h5_typename(field.type.resolve())
            is_static = field.is_static
            if field == tab.key:
                is_used = False
            else:
                is_used = True
            fields.append(
                Field(name, print_name, dataset_name, type, h5_type, is_static, is_used)
            )

        contagions = []
        for contagion in tab.contagions:
            name = mangle(contagion.name)
            state_type = typename(contagion.state_type.resolve())
            contagions.append((name, state_type))

        return cls(fields, contagions, tab.key.name)


@attrs.define
class EdgeTable:
    fields: list[Field]
    contagions: list[str]  # contagion name
    target_node_key: str
    source_node_key: str

    @classmethod
    def make(cls, tab: ast1.EdgeTable) -> EdgeTable:
        fields = []
        for field in tab.fields:
            name = mangle(field.name)
            print_name = field.name
            dataset_name = f"/edge/{field.name}"
            type = typename(field.type.resolve())
            h5_type = h5_typename(field.type.resolve())
            is_static = field.is_static
            if field == tab.target_node_key or field == tab.source_node_key:
                is_used = False
            else:
                is_used = True
            fields.append(
                Field(name, print_name, dataset_name, type, h5_type, is_static, is_used)
            )

        contagions = []
        for contagion in tab.contagions:
            contagions.append(mangle(contagion.name))

        return cls(
            fields, contagions, tab.target_node_key.name, tab.source_node_key.name
        )


@attrs.define
class ContagionOutput:
    name: str  # contagion name
    print_name: str  # unmangled name for datasets
    state_type: str
    state_h5_type: str

    @classmethod
    def make(cls, c: ast1.Contagion) -> ContagionOutput:
        name = mangle(c.name)
        print_name = c.name
        state_type = typename(c.state_type.resolve())
        state_h5_type = h5_typename(c.state_type.resolve())
        return cls(name, print_name, state_type, state_h5_type)


@attrs.define
class ConstantDist:
    name: str
    v: str

    @classmethod
    def make(cls, d: ast1.ConstantDist) -> ConstantDist:
        return cls(
            name=mangle(d.name),
            v=str(d.v),
        )


@attrs.define
class DiscreteDist:
    name: str
    probs: list[str]
    alias: list[str]
    vs: list[str]

    @classmethod
    def make(cls, d: ast1.DiscreteDist) -> DiscreteDist:
        name = mangle(d.name)
        table = AliasTable.make(d.ps)
        probs = [str(p) for p in table.probs]
        alias = [str(p) for p in table.alias]
        vs = [str(v) for v in d.vs]
        return cls(name, probs, alias, vs)


@attrs.define
class NormalDist:
    name: str
    mean: str
    std: str
    min: str
    max: str

    @classmethod
    def make(cls, d: ast1.NormalDist) -> NormalDist:
        return cls(
            name=mangle(d.name),
            mean=str(d.mean),
            std=str(d.std),
            min=str(d.min),
            max=str(d.max),
        )


@attrs.define
class UniformDist:
    name: str
    low: str
    high: str

    @classmethod
    def make(cls, d: ast1.UniformDist) -> UniformDist:
        return cls(
            name=mangle(d.name),
            low=str(d.low),
            high=str(d.high),
        )


@attrs.define
class Function:
    name: str
    params: list[tuple[str, str]]
    return_: str
    variables: list[tuple[str, str]]
    body: str
    line: str

    @classmethod
    def make(cls, name: str, x: ast1.Function) -> Function:
        params = []
        for p in x.params:
            p_name = ref_str(p)
            p_type = typename(p.type.resolve())
            params.append((p_name, p_type))

        if x.return_ is None:
            return_ = "void"
        else:
            return_ = typename(x.return_.resolve())

        variables = []
        for v in x.variables():
            v_name = ref_str(v)
            v_type = typename(v.type.resolve())
            variables.append((v_name, v_type))

        body = IndentedLines(cur_indent=1)
        for s in x.body:
            body.extend(statement_lines(s, body.cur_indent + 1))
        body = body.to_string()

        line = make_line(x.pos)
        return cls(name, params, return_, variables, body, line)


@attrs.define
class SingleEdgeTransition:
    entry: str
    exit: str
    dwell_dist: str

    @classmethod
    def make(cls, t: ast1.Transition) -> SingleEdgeTransition:
        entry = ref_str(t.entry.resolve())
        exit = ref_str(t.exit.resolve())
        dwell_dist = mangle(t.dwell.resolve().name)
        return cls(entry, exit, dwell_dist)


@attrs.define
class MultiEdgeTransition:
    entry: str
    probs: list[str]
    alias: list[str]
    exits: list[str]
    dwell_dists: list[str]

    @classmethod
    def make(cls, ts: list[ast1.Transition]) -> MultiEdgeTransition:
        entry = ref_str(ts[0].entry.resolve())

        ps = [t.p for t in ts]
        table = AliasTable.make(ps)
        probs = [str(p) for p in table.probs]
        alias = [str(p) for p in table.alias]

        exits = [ref_str(t.exit.resolve()) for t in ts]
        dwell_dists = [mangle(t.dwell.resolve().name) for t in ts]
        return cls(entry, probs, alias, exits, dwell_dists)


@attrs.define
class Transition:
    single: list[SingleEdgeTransition]
    multi: list[MultiEdgeTransition]
    notrans: list[str]

    @classmethod
    def make(cls, c: ast1.Contagion) -> Transition:
        entry_transi = defaultdict(list)
        for transi in c.transitions:
            entry_transi[ref_str(transi.entry.resolve())].append(transi)

        single: list[SingleEdgeTransition] = []
        multi: list[MultiEdgeTransition] = []
        for vs in entry_transi.values():
            if len(vs) == 1:
                single.append(SingleEdgeTransition.make(vs[0]))
            else:
                multi.append(MultiEdgeTransition.make(vs))

        notrans = []
        for const in c.state_type.resolve().consts:
            const = mangle(const)
            if const not in entry_transi:
                notrans.append(const)

        return cls(
            single=single,
            multi=multi,
            notrans=notrans,
        )


@attrs.define
class Transmission:
    transms: list[
        tuple[str, list[tuple[str, list[str]]]]
    ]  # entry -> [exit -> [contact]]
    susceptibility: str
    infectivity: str
    transmissibility: str
    enabled: str

    @classmethod
    def make(cls, c: ast1.Contagion) -> Transmission:
        transms = defaultdict(lambda: defaultdict(list))
        for t in c.transmissions:
            contact = ref_str(t.contact.resolve())
            entry = ref_str(t.entry.resolve())
            exit = ref_str(t.exit.resolve())
            transms[entry][exit].append(contact)
        transms = [
            (entry, [(exit, contacts) for exit, contacts in xs.items()])
            for entry, xs in transms.items()
        ]

        susceptibility = call_str(c.susceptibility.resolve())
        infectivity = call_str(c.infectivity.resolve())
        transmissibility = call_str(c.transmissibility.resolve())
        enabled = call_str(c.enabled.resolve())

        return cls(
            transms=transms,
            susceptibility=susceptibility,
            infectivity=infectivity,
            transmissibility=transmissibility,
            enabled=enabled,
        )


@attrs.define
class Contagion:
    name: str
    print_name: str
    state_type: str
    num_states: int
    transition: Transition
    transmission: Transmission

    @classmethod
    def make(cls, c: ast1.Contagion) -> Contagion:
        name = mangle(c.name)
        print_name = c.name
        state_type = typename(c.state_type.resolve())
        num_states = len(c.state_type.resolve().consts)
        transition = Transition.make(c)
        transmission = Transmission.make(c)
        return cls(name, print_name, state_type, num_states, transition, transmission)


@attrs.define
class SelectUsing:
    name: str
    set_name: str
    function_name: str

    @classmethod
    def make(cls, s: ast1.SelectUsing) -> SelectUsing:
        name = s.name
        set_name = ref_str(s.set.resolve())
        function_name = call_str(s.function.resolve())
        return cls(name, set_name, function_name)


@attrs.define
class SelectApprox:
    name: str
    set_name: str
    amount: str
    parent_set_name: str

    @classmethod
    def make(cls, s: ast1.SelectApprox) -> SelectApprox:
        name = s.name
        set_name = ref_str(s.set.resolve())
        amount = str(s.amount)
        parent_set_name = ref_str(s.parent.resolve())
        return cls(name, set_name, amount, parent_set_name)


@attrs.define
class SelectRelative:
    name: str
    set_name: str
    amount: str
    parent_set_name: str

    @classmethod
    def make(cls, s: ast1.SelectRelative) -> SelectRelative:
        name = s.name
        set_name = ref_str(s.set.resolve())
        amount = str(s.amount)
        parent_set_name = ref_str(s.parent.resolve())
        return cls(name, set_name, amount, parent_set_name)


@attrs.define
class ForeachStatement:
    name: str
    set_name: str
    function_name: str

    @classmethod
    def make(cls, s: ast1.ForeachStatement) -> ForeachStatement:
        name = s.name
        set_name = ref_str(s.set.resolve())
        function_name = call_str(s.function.resolve())
        return cls(name, set_name, function_name)


@attrs.define
class Source:
    module: str
    enums: list[EnumType]
    configs: list[Config]
    globals: list[Global]
    node_table: NodeTable
    edge_table: EdgeTable
    contagion_outputs: list[ContagionOutput]
    nodesets: list[str]
    edgesets: list[str]
    constant_dists: list[ConstantDist]
    discrete_dists: list[DiscreteDist]
    uniform_dists: list[UniformDist]
    normal_dists: list[NormalDist]
    functions: list[Function]
    contagions: list[Contagion]
    select_using_node: list[SelectUsing]
    select_using_edge: list[SelectUsing]
    select_approx_node: list[SelectApprox]
    select_approx_edge: list[SelectApprox]
    select_relative_node: list[SelectRelative]
    select_relative_edge: list[SelectRelative]
    foreach_node_statement: list[ForeachStatement]
    foreach_edge_statement: list[ForeachStatement]

    @classmethod
    def make(cls, source: ast1.Source) -> Source:
        module = source.module

        enums = [EnumType.make(e) for e in source.enums]
        configs = [Config.make(e) for e in source.configs]
        globals = [Global.make(e) for e in source.globals]
        node_table = NodeTable.make(source.node_table)
        edge_table = EdgeTable.make(source.edge_table)
        contagion_outputs = [ContagionOutput.make(c) for c in source.contagions]
        nodesets = [ref_str(s) for s in source.nodesets]
        edgesets = [ref_str(s) for s in source.edgesets]

        constant_dists: list[ConstantDist] = []
        discrete_dists: list[DiscreteDist] = []
        uniform_dists: list[UniformDist] = []
        normal_dists: list[NormalDist] = []
        for dist in source.distributions:
            match dist:
                case ast1.ConstantDist():
                    constant_dists.append(ConstantDist.make(dist))
                case ast1.DiscreteDist():
                    discrete_dists.append(DiscreteDist.make(dist))
                case ast1.UniformDist():
                    uniform_dists.append(UniformDist.make(dist))
                case ast1.NormalDist():
                    normal_dists.append(NormalDist.make(dist))
                case _ as unexpected:
                    assert_never(unexpected)

        functions: list[Function] = []
        for fn in source.functions:
            name = call_str(fn)
            functions.append(Function.make(name, fn))
        functions.append(Function.make("do_initialize", source.initialize))
        functions.append(Function.make("do_intervene", source.intervene))

        contagions = [Contagion.make(c) for c in source.contagions]

        select_using_node: list[SelectUsing] = []
        select_using_edge: list[SelectUsing] = []
        select_approx_node: list[SelectApprox] = []
        select_approx_edge: list[SelectApprox] = []
        select_relative_node: list[SelectRelative] = []
        select_relative_edge: list[SelectRelative] = []
        foreach_node_statement: list[ForeachStatement] = []
        foreach_edge_statement: list[ForeachStatement] = []
        for fn in chain(source.functions, [source.initialize, source.intervene]):
            for stmt in fn.device_statements():
                set = stmt.set.resolve()
                match [stmt, set]:
                    case [ast1.SelectUsing() as s, ast1.NodeSet()]:
                        select_using_node.append(SelectUsing.make(s))
                    case [ast1.SelectUsing() as s, ast1.EdgeSet()]:
                        select_using_edge.append(SelectUsing.make(s))
                    case [ast1.SelectApprox() as s, ast1.NodeSet()]:
                        select_approx_node.append(SelectApprox.make(s))
                    case [ast1.SelectApprox() as s, ast1.EdgeSet()]:
                        select_approx_edge.append(SelectApprox.make(s))
                    case [ast1.SelectRelative() as s, ast1.NodeSet()]:
                        select_relative_node.append(SelectRelative.make(s))
                    case [ast1.SelectRelative() as s, ast1.EdgeSet()]:
                        select_relative_edge.append(SelectRelative.make(s))
                    case [ast1.ForeachStatement() as s, ast1.NodeSet()]:
                        foreach_node_statement.append(ForeachStatement.make(s))
                    case [ast1.ForeachStatement() as s, ast1.EdgeSet()]:
                        foreach_edge_statement.append(ForeachStatement.make(s))
                    case _ as unexpected:
                        raise UnexpectedValue(unexpected, stmt.pos)

        return cls(
            module=module,
            enums=enums,
            configs=configs,
            globals=globals,
            node_table=node_table,
            edge_table=edge_table,
            contagion_outputs=contagion_outputs,
            nodesets=nodesets,
            edgesets=edgesets,
            constant_dists=constant_dists,
            discrete_dists=discrete_dists,
            uniform_dists=uniform_dists,
            normal_dists=normal_dists,
            functions=functions,
            contagions=contagions,
            select_using_node=select_using_node,
            select_using_edge=select_using_edge,
            select_approx_node=select_approx_node,
            select_approx_edge=select_approx_edge,
            select_relative_node=select_relative_node,
            select_relative_edge=select_relative_edge,
            foreach_node_statement=foreach_node_statement,
            foreach_edge_statement=foreach_edge_statement,
        )


def do_prepare(output: Path | None, input: Path) -> Path:
    input_bytes = input.read_bytes()
    pt = mk_pt(str(input), input_bytes)
    ast1 = mk_ast1(input, pt)

    if output is None:
        output = input.parent
    output.mkdir(parents=True, exist_ok=True)

    source = Source.make(ast1)

    with open(output / "CMakeLists.txt", "wt") as fobj:
        template = ENVIRONMENT.get_template("cmakelists_txt_cpu.jinja2")
        fobj.write(template.render(source=source, input=str(input.absolute())))

    return output


def do_compile(output: Path | None, input: Path) -> Path:
    input_bytes = input.read_bytes()
    pt = mk_pt(str(input), input_bytes)
    ast1 = mk_ast1(input, pt)

    if output is None:
        output = input.parent
    output.mkdir(parents=True, exist_ok=True)

    source = Source.make(ast1)

    with open(output / "simulator.cpp", "wt") as fobj:
        template = ENVIRONMENT.get_template("simulator_cpp_cpu.jinja2")
        fobj.write(template.render(source=source))

    with open(output / "sim_utils.h", "wt") as fobj:
        fobj.write(STATIC_DIR.joinpath("sim_utils.h").read_text())

    return output


@click.command()
@click.argument(
    "input",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
)
def print_cpu_ir(input: Path):
    """Print intermediate representation."""
    input_bytes = input.read_bytes()

    try:
        pt = mk_pt(str(input), input_bytes)
        ast1 = mk_ast1(input, pt)
        source = Source.make(ast1)
        rich.print(source)
    except (ParseTreeConstructionError, ASTConstructionError, CodegenError) as e:
        e.rich_print()
        sys.exit(1)


@click.group()
def codegen_cpu():
    """Generate code targeted for CPUs."""


@codegen_cpu.command()
@click.option(
    "-o",
    "--output",
    type=click.Path(exists=False, file_okay=False, dir_okay=True, path_type=Path),
    help="Output directory.",
)
@click.argument(
    "input",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
)
def prepare(output: Path | None, input: Path):
    """Generate CMakeLists.txt."""
    try:
        do_prepare(output, input)
    except (ParseTreeConstructionError, ASTConstructionError) as e:
        e.rich_print()
        sys.exit(1)


@codegen_cpu.command()
@click.option(
    "-o",
    "--output",
    type=click.Path(exists=False, file_okay=False, dir_okay=True, path_type=Path),
    help="Output directory.",
)
@click.argument(
    "input",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
)
def compile(output: Path | None, input: Path):
    """Compile simulator code."""
    try:
        do_compile(output, input)
    except (ParseTreeConstructionError, ASTConstructionError) as e:
        e.rich_print()
        sys.exit(1)


def pprint_cmd(cmd):
    cmd = dedent(cmd).strip()
    cmd = Syntax(cmd, "bash", theme="gruvbox-dark")
    rich.print(cmd)


def verbose_run(cmd, *args, **kwargs):
    pprint_cmd(cmd)
    cmd = shlex.split(cmd)
    return do_run(cmd, *args, **kwargs)


@codegen_cpu.command()
@click.argument(
    "input",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
)
def run(input: Path):
    """Build and run simulator."""
    try:
        with TemporaryDirectory(
            prefix=f"{input.stem}-", suffix="-episim37"
        ) as temp_output_dir:
            output = Path(temp_output_dir).absolute()
            rich.print(f"[cyan]Temp output dir:[/cyan] {output!s}")

            rich.print("[cyan]Creating CmakeList.txt[/cyan]")
            output = do_prepare(output, input)

            rich.print("[cyan]Running cmake[/cyan]")
            cmd = f"cmake -S . -B '{output!s}/build'"
            verbose_run(cmd, cwd=output)

            rich.print("[cyan]Running make[/cyan]")
            cmd = f"make -C '{output!s}/build'"
            verbose_run(cmd)

            rich.print("[cyan]Running simulator[/cyan]")
            cmd = f"'{output!s}/build/simulator'"
            verbose_run(cmd)
    except (ParseTreeConstructionError, ASTConstructionError, CodegenError) as e:
        e.rich_print()
        sys.exit(1)
