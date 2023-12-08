"""Intermediate Representation v1."""

from __future__ import annotations

from pathlib import Path
from typing import assert_never
from collections import defaultdict

import attrs
import click
import rich
import rich.markup

from . import ast1
from .ast1 import mk_ast1, ASTConstructionError
from .parse_tree import mk_pt, ParseTreeConstructionError, SourcePosition
from .alias_table import AliasTable


class IRConstructionError(Exception):
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


Error = IRConstructionError


class UnexpectedValue(IRConstructionError):
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

        "node":  "node_type",
        "edge":  "edge_type",
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

    return "LOCAL_H5_TYPE_" + t.upper()


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
        case ("float_type", "float" | "double"):
            return "std::stod"
        case _ as unexpected:
            raise ValueError(unexpected)


def make_line(pos: SourcePosition | None) -> str:
    if pos is None:
        return ""

    return f'#line {pos.line} "{pos.source}"'


def mangle(*args: str) -> str:
    if len(args) == 1:
        return "_" + args[0]

    return "_" + "__".join(args)


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


def ref(x: ast1.Referable) -> str:
    match x:
        case ast1.EnumConstant():
            return mangle(x.name)
        case ast1.Config():
            return mangle(x.name)
        case ast1.Param():
            return mangle(x.name)
        case ast1.Variable():
            return mangle(x.var)
        case ast1.BuiltinFunction():
            return mangle(x.name)
        case ast1.Function():
            return mangle(x.name)
        case [ast1.Contagion() as c, ast1.BuiltinFunction() as f]:
            return mangle(c.name, f.name)
        case [ast1.Contagion() as c, ast1.Function() as f]:
            return mangle(c.name, f.name)
        case [ast1.Param() | ast1.Variable() as n, ast1.NodeField() as f]:
            n_ref = ref(n)
            f_ref = mangle(f.name)
            return f"{n_ref}.n->{f_ref}[{n_ref}.i]"
        case [ast1.Param() | ast1.Variable() as e, ast1.EdgeField() as f]:
            e_ref = ref(e)
            f_ref = mangle(f.name)
            return f"{e_ref}.e->{f_ref}[{e_ref}.i]"
        case [
            ast1.Param() | ast1.Variable() as n,
            ast1.Contagion() as c,
            ast1.NodeField() as f,
        ]:
            n_ref = ref(n)
            f_ref = mangle(c.name, f.name)
            return f"{n_ref}.n->{f_ref}[{n_ref}.i]"
        case [
            ast1.Param() | ast1.Variable() as e,
            ast1.SourceNodeAccessor(),
        ]:
            e_ref = ref(e)
            return f"node_type({e_ref}.n, {e_ref}.e->source_node_index[{e_ref}.i])"
        case [
            ast1.Param() | ast1.Variable() as e,
            ast1.TargetNodeAccessor(),
        ]:
            e_ref = ref(e)
            return f"node_type({e_ref}.n, {e_ref}.e->target_node_index[{e_ref}.i])"
        case [
            ast1.Param() | ast1.Variable() as e,
            ast1.SourceNodeAccessor(),
            ast1.NodeField() as f,
        ]:
            e_ref = ref(e)
            f_ref = mangle(f.name)
            return f"{e_ref}.n->{f_ref}[{e_ref}.e->source_node_index[{e_ref}.i]]"
        case [
            ast1.Param() | ast1.Variable() as e,
            ast1.SourceNodeAccessor(),
            ast1.Contagion() as c,
            ast1.NodeField() as f,
        ]:
            e_ref = ref(e)
            f_ref = mangle(c.name, f.name)
            return f"{e_ref}.n->{f_ref}[{e_ref}.e->source_node_index[{e_ref}.i]]"
        case [
            ast1.Param() | ast1.Variable() as e,
            ast1.TargetNodeAccessor(),
            ast1.NodeField() as f,
        ]:
            e_ref = ref(e)
            f_ref = mangle(f.name)
            return f"{e_ref}.n->{f_ref}[{e_ref}.e->target_node_index[{e_ref}.i]]"
        case [
            ast1.Param() | ast1.Variable() as e,
            ast1.TargetNodeAccessor(),
            ast1.Contagion() as c,
            ast1.NodeField() as f,
        ]:
            e_ref = ref(e)
            f_ref = mangle(c.name, f.name)
            return f"{e_ref}.n->{f_ref}[{e_ref}.e->target_node_index[{e_ref}.i]]"
        case _ as unexpected:
            assert_never(unexpected)


@attrs.define
class EnumType:
    name: str
    base_type: str
    consts: list[str]
    line: str

    @classmethod
    def make(cls, et: ast1.EnumType) -> EnumType:
        name = typename(et)
        base_type = enum_base_type(et)
        consts = [mangle(c) for c in et.consts]
        line = make_line(et.pos)
        return cls(name, base_type, consts, line)


@attrs.define
class Config:
    name: str
    type: str
    from_str_fn: str
    env_var: str
    print_name: str
    default: str
    line: str

    @classmethod
    def make(cls, c: ast1.Config) -> Config:
        name = ref(c)
        type = typename(c.type.resolve())
        from_str_fn = cstr_to_ctype_fn(type)
        env_var = c.name.upper()
        default = expression_str(c.default)
        line = make_line(c.pos)
        return cls(name, type, from_str_fn, env_var, c.name, default, line)


@attrs.define
class Variable:
    var: str
    type: str
    init: str
    line: str

    @classmethod
    def make(cls, v: ast1.Variable) -> Variable:
        var = ref(v)
        type = typename(v.type.resolve())
        init = expression_str(v.init)
        line = make_line(v.pos)
        return cls(var, type, init, line)


@attrs.define
class Field:
    name: str
    dataset_name: str
    type: str
    h5_type: str
    is_static: bool
    line: str


@attrs.define
class StaticField:
    dataset_name: str
    type: str
    h5_type: str


@attrs.define
class CsrIncMatrix:
    indptr: StaticField

    @classmethod
    def make(cls) -> CsrIncMatrix:
        return cls(
            indptr=StaticField(
                "/incoming/incidence/csr/indptr",
                "edge_index_type",
                hdf5_type("edge_index_type"),
            ),
        )


@attrs.define
class NodeTable:
    fields: list[Field]
    line: str

    @classmethod
    def make(cls, tab: ast1.NodeTable) -> NodeTable:
        fields = []
        for field in tab.fields:
            name = mangle(field.name)
            dataset_name = f"/node/{field.name}"
            type = typename(field.type.resolve())
            h5_type = h5_typename(field.type.resolve())
            is_static = field.is_static
            line = make_line(field.pos)
            fields.append(Field(name, dataset_name, type, h5_type, is_static, line))

        for contagion in tab.contagions:
            for field in [contagion.state, contagion.next_state, contagion.dwell_time]:
                name = mangle(contagion.name, field.name)
                dataset_name = f"/node/{contagion.name}/{field.name}"
                type = typename(field.type.resolve())
                h5_type = h5_typename(field.type.resolve())
                is_static = field.is_static
                line = make_line(field.pos)
                fields.append(Field(name, dataset_name, type, h5_type, is_static, line))

        line = make_line(tab.pos)
        return cls(fields, line)


@attrs.define
class EdgeTable:
    fields: list[Field]
    target_node_index: StaticField
    source_node_index: StaticField
    line: str

    @classmethod
    def make(cls, tab: ast1.EdgeTable) -> EdgeTable:
        fields = []
        for field in tab.fields:
            name = mangle(field.name)
            dataset_name = f"/edge/{field.name}"
            type = typename(field.type.resolve())
            h5_type = h5_typename(field.type.resolve())
            is_static = field.is_static
            line = make_line(field.pos)
            fields.append(Field(name, dataset_name, type, h5_type, is_static, line))

        for c in tab.contagions:
            name = mangle(c.name) + "_transmission_prob"
            dataset_name = f"/edge/{name}"
            type = "float_type"
            h5_type = hdf5_type("float_type")
            is_static = False
            line = make_line(None)
            fields.append(Field(name, dataset_name, type, h5_type, is_static, line))

        target_node_index = StaticField(
            dataset_name="/edge/_target_node_index",
            type="node_index_type",
            h5_type=hdf5_type("node_index_type"),
        )
        source_node_index = StaticField(
            dataset_name="/edge/_source_node_index",
            type="node_index_type",
            h5_type=hdf5_type("node_index_type"),
        )

        line = make_line(tab.pos)
        return cls(fields, target_node_index, source_node_index, line)


@attrs.define
class ContagionOutput:
    name: str
    state_type: str
    state_h5_type: str
    gvar_name: str

    @classmethod
    def make(cls, c: ast1.Contagion) -> ContagionOutput:
        name = mangle(c.name) + "_output_type"
        state_type = typename(c.state_type.resolve())
        state_h5_type = h5_typename(c.state_type.resolve())
        gvar_name = mangle(c.name) + "_OUTPUT"
        return cls(name, state_type, state_h5_type, gvar_name)


@attrs.define
class ConstantDist:
    name: str
    v: str
    line: str

    @classmethod
    def make(cls, d: ast1.ConstantDist) -> ConstantDist:
        return cls(
            name=mangle(d.name),
            v=str(d.v),
            line=make_line(d.pos),
        )


@attrs.define
class DiscreteDist:
    name: str
    probs: list[str]
    alias: list[str]
    vs: list[str]
    line: str

    @classmethod
    def make(cls, d: ast1.DiscreteDist) -> DiscreteDist:
        name = mangle(d.name)
        table = AliasTable.make(d.ps)
        probs = [str(p) for p in table.probs]
        alias = [str(p) for p in table.alias]
        vs = [str(v) for v in d.vs]
        line = make_line(d.pos)
        return cls(name, probs, alias, vs, line)


@attrs.define
class NormalDist:
    name: str
    mean: str
    std: str
    min: str
    max: str
    line: str

    @classmethod
    def make(cls, d: ast1.NormalDist) -> NormalDist:
        return cls(
            name=mangle(d.name),
            mean=str(d.mean),
            std=str(d.std),
            min=str(d.min),
            max=str(d.max),
            line=make_line(d.pos),
        )


@attrs.define
class UniformDist:
    name: str
    low: str
    high: str
    line: str

    @classmethod
    def make(cls, d: ast1.UniformDist) -> UniformDist:
        return cls(
            name=mangle(d.name),
            low=str(d.low),
            high=str(d.high),
            line=make_line(d.pos),
        )


@attrs.define
class SingleEdgeTransition:
    entry: str
    exit: str
    dwell_dist: str

    @classmethod
    def make(cls, t: ast1.Transition) -> SingleEdgeTransition:
        entry = ref(t.entry.resolve())
        exit = ref(t.exit.resolve())
        dwell_dist = mangle(t.dwell.resolve().name)
        return cls(entry, exit, dwell_dist)


@attrs.define
class MultiEdgeTransition:
    name: str
    state_type: str
    entry: str
    probs: list[str]
    alias: list[str]
    exits: list[str]
    dwell_dists: list[str]

    @classmethod
    def make(
        cls, parent_name: str, state_type: str, ts: list[ast1.Transition]
    ) -> MultiEdgeTransition:
        entry = ref(ts[0].entry.resolve())
        name = parent_name + f"_{entry}"

        ps = [t.p for t in ts]
        table = AliasTable.make(ps)
        probs = [str(p) for p in table.probs]
        alias = [str(p) for p in table.alias]

        exits = [ref(t.exit.resolve()) for t in ts]
        dwell_dists = [mangle(t.dwell.resolve().name) for t in ts]
        return cls(name, state_type, entry, probs, alias, exits, dwell_dists)


@attrs.define
class DoTransition:
    name: str
    state_type: str
    output_type: str
    output_gvar: str
    state: str
    next_state: str
    dwell_time: str
    single: list[tuple[str, SingleEdgeTransition]]  # entry -> transition
    multi: list[tuple[str, MultiEdgeTransition]]  # entry -> transition
    notrans: list[str]  # entry

    @classmethod
    def make(cls, c: ast1.Contagion, o: ContagionOutput) -> DoTransition:
        name = ref((c, c.do_transition))
        state_type = typename(c.state_type.resolve())

        entry_transi = defaultdict(list)
        for transi in c.transitions:
            entry_transi[ref(transi.entry.resolve())].append(transi)

        single = []
        multi = []
        for vs in entry_transi.values():
            if len(vs) == 1:
                single.append(SingleEdgeTransition.make(vs[0]))
            else:
                multi.append(MultiEdgeTransition.make(name, state_type, vs))

        notrans = []
        for const in c.state_type.resolve().consts:
            const = mangle(const)
            if const not in entry_transi:
                notrans.append(const)

        state = mangle(c.name, c.state.name)
        next_state = mangle(c.name, c.next_state.name)
        dwell_time = mangle(c.name, c.dwell_time.name)
        return cls(
            name=name,
            state_type=state_type,
            output_type=o.name,
            output_gvar=o.gvar_name,
            state=state,
            next_state=next_state,
            dwell_time=dwell_time,
            single=single,
            multi=multi,
            notrans=notrans,
        )


@attrs.define
class DoTransmission:
    name: str
    state_type: str
    output_type: str
    output_gvar: str
    state: str
    tprob: str
    transms: list[tuple[str, list[tuple[str, str]]]]  # entry -> [contact, exit]
    susceptibility: str
    infectivity: str
    transmissibility: str
    enabled: str

    @classmethod
    def make(cls, c: ast1.Contagion, o: ContagionOutput):
        name = ref((c, c.do_transmission))
        state_type = typename(c.state_type.resolve())

        transms = defaultdict(list)
        for t in c.transmissions:
            contact = ref(t.contact.resolve())
            entry = ref(t.entry.resolve())
            exit = ref(t.exit.resolve())
            transms[entry].append((contact, exit))
        transms = list(transms.items())

        susceptibility = ref((c, c.susceptibility))
        infectivity = ref((c, c.infectivity))
        transmissibility = ref((c, c.transmissibility))
        enabled = ref((c, c.enabled))

        state = mangle(c.name, c.state.name)
        tprob = mangle(c.name) + "_transmission_prob"
        return cls(
            name=name,
            state_type=state_type,
            output_type=o.name,
            output_gvar=o.gvar_name,
            state=state,
            tprob=tprob,
            transms=transms,
            susceptibility=susceptibility,
            infectivity=infectivity,
            transmissibility=transmissibility,
            enabled=enabled,
        )


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
            return ref(e.resolve())
        case ast1.FunctionCall():
            # TODO: Add check that this actually resolves to a function
            # TODO: Add check that parameters are correct
            function = ref(e.function.resolve())
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
            left = expression_str(s.left)
            op = s.operator
            right = expression_str(s.right)
            lines.append(f"{left} {op} {right};")
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
            left = ref(s)
            right = expression_str(s.init)
            lines.append(f"{left} = {right};")
        case ast1.NodeSet():
            pass
        case ast1.EdgeSet():
            pass
        case ast1.UpdateSet():
            pass
        case ast1.ForeachLoop():
            pass
        case _ as unexpected:
            assert_never(unexpected)

    return lines


@attrs.define
class Function:
    name: str
    params: list[tuple[str, str]]
    return_: str
    variables: list[tuple[str, str]]
    nodesets: list[str]
    edgesets: list[str]
    body: str
    line: str

    @classmethod
    def make(cls, name: str, x: ast1.Function) -> Function:
        params = []
        for p in x.params:
            p_name = ref(p)
            p_type = typename(p.type.resolve())
            params.append((p_name, p_type))

        if x.return_ is None:
            return_ = "void"
        else:
            return_ = typename(x.return_.resolve())

        variables = []
        for v in x.variables:
            v_name = ref(v)
            v_type = typename(v.type.resolve())
            variables.append((v_name, v_type))

        nodesets = []
        for v in x.nodesets:
            nodesets.append(mangle(v.name))

        edgesets = []
        for v in x.edgesets:
            edgesets.append(mangle(v.name))

        body = IndentedLines(cur_indent=1)
        for s in x.body:
            body.extend(statement_lines(s, body.cur_indent + 1))
        body = body.to_string()

        line = make_line(x.pos)
        return cls(name, params, return_, variables, nodesets, edgesets, body, line)


@attrs.define
class DeferredType:
    name: str
    base_ctype: str
    h5_type: str
    base_h5_type: str


DEFERRED_TYPES = {
    "int_type": "int64_t",
    "uint_type": "uint64_t",
    "float_type": "double",
    "bool_type": "uint8_t",
    "size_type": "uint64_t",
    "node_index_type": "uint32_t",
    "edge_index_type": "uint64_t",
}


@attrs.define
class Source:
    module: str
    deferred_types: list[DeferredType]
    enums: list[EnumType]
    configs: list[Config]
    variables: list[Variable]
    node_table: NodeTable
    edge_table: EdgeTable
    in_inc_csr: CsrIncMatrix
    contagion_outputs: list[ContagionOutput]
    constant_dists: list[ConstantDist]
    discrete_dists: list[DiscreteDist]
    uniform_dists: list[UniformDist]
    normal_dists: list[NormalDist]
    do_transitions: list[DoTransition]
    do_transmissions: list[DoTransmission]
    functions: list[Function]

    @classmethod
    def make(cls, x: ast1.Source) -> Source:
        module = x.module

        deferred_types = []
        for type, base_ctype in DEFERRED_TYPES.items():
            deferred_types.append(
                DeferredType(type, base_ctype, hdf5_type(type), hdf5_type(base_ctype))
            )

        enums = [EnumType.make(e) for e in x.enums]
        configs = [Config.make(c) for c in x.configs]
        variables = [Variable.make(v) for v in x.variables]
        node_table = NodeTable.make(x.node_table)
        edge_table = EdgeTable.make(x.edge_table)
        in_inc_csr = CsrIncMatrix.make()

        constant_dists: list[ConstantDist] = []
        discrete_dists: list[DiscreteDist] = []
        uniform_dists: list[UniformDist] = []
        normal_dists: list[NormalDist] = []
        for dist in x.distributions:
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

        contagion_outputs = [ContagionOutput.make(c) for c in x.contagions]
        do_transitions = [
            DoTransition.make(c, o) for c, o in zip(x.contagions, contagion_outputs)
        ]
        do_transmissions = [
            DoTransmission.make(c, o) for c, o in zip(x.contagions, contagion_outputs)
        ]

        functions: list[Function] = []
        for c in x.contagions:
            name = ref((c, c.susceptibility))
            functions.append(Function.make(name, c.susceptibility))

            name = ref((c, c.infectivity))
            functions.append(Function.make(name, c.infectivity))

            name = ref((c, c.transmissibility))
            functions.append(Function.make(name, c.transmissibility))

            name = ref((c, c.enabled))
            functions.append(Function.make(name, c.enabled))

        for f in x.functions:
            name = ref(f)
            functions.append(Function.make(name, f))

        functions.append(Function.make("do_main", x.main_function))

        return cls(
            module=module,
            deferred_types=deferred_types,
            enums=enums,
            configs=configs,
            variables=variables,
            node_table=node_table,
            edge_table=edge_table,
            in_inc_csr=in_inc_csr,
            contagion_outputs=contagion_outputs,
            constant_dists=constant_dists,
            discrete_dists=discrete_dists,
            uniform_dists=uniform_dists,
            normal_dists=normal_dists,
            do_transitions=do_transitions,
            do_transmissions=do_transmissions,
            functions=functions,
        )


def mk_ir1(source: ast1.Source) -> Source:
    """Make ir1 from ast1."""
    return Source.make(source)


@click.command()
@click.argument(
    "filename",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
)
def print_ir1(filename: Path):
    """Print the IR1."""
    file_bytes = filename.read_bytes()
    try:
        pt = mk_pt(str(filename), file_bytes)
        ast1 = mk_ast1(filename, pt)
        ir1 = mk_ir1(ast1)
        rich.print(ir1)
    except (ParseTreeConstructionError, ASTConstructionError, IRConstructionError) as e:
        e.rich_print()
