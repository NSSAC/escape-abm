"""Abstract syntax tree v1."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import attrs
import click

import rich
import rich.markup

from .parse_tree import mk_pt, PTNode, ParseTreeConstructionError, SourcePosition


class ASTConstructionError(Exception):
    def __init__(self, type: str, explanation: str, pos: SourcePosition | None):
        super().__init__()
        self.type = type
        self.explanation = explanation
        self.pos = pos

    def __str__(self):
        return "Failed to construct AST"

    def rich_print(self):
        rich.print(f"[red]{self}[/red]")

        if self.pos is not None:
            etype = f"[red]{self.type}[/red]"
            fpath = f"[yellow]{self.pos.source}[/yellow]"
            line = self.pos.line
            col = self.pos.col
            expl = rich.markup.escape(self.explanation)

            rich.print(f"{etype}:{fpath}:{line}:{col}:{expl}")
            print(self.pos.text)
        else:
            print(self.explanation)


class UnexpectedValue(ASTConstructionError):
    def __init__(self, unexpected, pos: SourcePosition):
        super().__init__("Unexpected value", repr(unexpected), pos)


Error = ASTConstructionError


@attrs.define
class Scope:
    name: str
    names: dict[str, Any] = attrs.field(factory=dict)
    parent: Scope | None = attrs.field(default=None)

    def __rich_repr__(self):
        yield "name", self.name
        yield "names", {k: type(v).__name__ for k, v in self.names.items()}
        if self.parent is not None:
            yield "parent", self.parent.name

    def resolve(self, k: str, pos: SourcePosition | None) -> Any:
        if k in self.names:
            return self.names[k]
        elif self.parent is not None:
            return self.parent.resolve(k, pos)

        raise Error("Undefined reference", f"{k} is not defined", pos)

    def define(self, k: str, v: Any, pos: SourcePosition | None):
        if k in self.names:
            raise Error(
                "Already defined",
                f"{k} has been already defined",
                pos,
            )

        self.names[k] = v


def smallest_int_type(max_val: int) -> str | None:
    if max_val < 2**8:
        return "u8"
    elif max_val < 2**16:
        return "u16"
    elif max_val < 2**32:
        return "u32"
    elif max_val < 2**32:
        return "u64"
    else:
        return None


def make_literal(n: PTNode, types: type) -> Any:
    match n.type:
        case "integer":
            v = int(n.text)
        case "float":
            v = float(n.text)
        case "boolean":
            v = n.text == True
        case "string":
            v = n.text
        case _ as unexpcted:
            raise UnexpectedValue(unexpcted, n.pos)

    if not isinstance(v, types):
        raise UnexpectedValue(v, n.pos)

    return v


@attrs.define
class BuiltinType:
    name: str


def add_builtin_type(type: str, scope: Scope, pos: SourcePosition):
    scope.define(type, BuiltinType(type), pos)


@attrs.define
class EnumConstant:
    name: str
    type: EnumType


@attrs.define
class ConstRef:
    const: str
    scope: Scope = attrs.field(repr=False, eq=False)
    pos: SourcePosition | None = attrs.field(default=None, repr=False, eq=False)

    def __str__(self):
        return self.const

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> ConstRef:
        type = node.text
        return ConstRef(type, scope, node.pos)

    def resolve(self) -> EnumConstant:
        v = self.scope.resolve(self.const, self.pos)
        if isinstance(v, EnumConstant):
            return v

        raise Error(
            "Invalid value",
            "Expected an enumeration constant",
            self.pos,
        )


@attrs.define
class EnumType:
    name: str
    base_type: BuiltinType
    consts: list[str]
    pos: SourcePosition | None = attrs.field(default=None, repr=False, eq=False)

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> EnumType:
        name = node.field("name").text
        consts = [child.text for child in node.fields("const")]

        base_type = smallest_int_type(len(consts) - 1)
        if base_type is None:
            raise Error("Large enum", "Enum is too big", node.pos)

        obj = cls(name, BuiltinType(base_type), consts, node.pos)
        for child in node.fields("const"):
            enum_const = EnumConstant(child.text, obj)
            scope.define(enum_const.name, enum_const, child.pos)

        scope.define(name, obj, node.pos)
        return obj


@attrs.define
class TypeRef:
    type: str
    scope: Scope = attrs.field(repr=False, eq=False)
    pos: SourcePosition | None = attrs.field(default=None, repr=False, eq=False)

    def __str__(self):
        return self.type

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> TypeRef:
        type = node.text
        return TypeRef(type, scope, node.pos)

    def resolve(self) -> BuiltinType | EnumType:
        v = self.scope.resolve(self.type, self.pos)
        if isinstance(v, BuiltinType | EnumType):
            return v

        raise Error(
            "Invalid type",
            "Expected a built in or enumerated type",
            self.pos,
        )

    def resolve_builtin(self) -> BuiltinType:
        v = self.scope.resolve(self.type, self.pos)
        if isinstance(v, BuiltinType):
            return v

        raise Error(
            "Invalid type",
            "Expected a built type",
            self.pos,
        )

    def resolve_enum(self) -> EnumType:
        v = self.scope.resolve(self.type, self.pos)
        if isinstance(v, EnumType):
            return v

        raise Error(
            "Invalid type",
            "Expected an enumerated type",
            self.pos,
        )


@attrs.define
class Config:
    name: str
    type: TypeRef | BuiltinType
    default: int | float | bool
    pos: SourcePosition | None = attrs.field(default=None, repr=False, eq=False)

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> Config:
        name = node.field("name").text
        type = TypeRef.make(node.field("type"), scope)
        default = make_literal(node.field("default"), int | float | bool)
        obj = cls(name, type, default, node.pos)
        scope.define(name, obj, node.pos)
        return obj

    def resolve(self):
        if isinstance(self.type, TypeRef):
            self.type = self.type.resolve_builtin()


@attrs.define
class Variable:
    var: str
    type: TypeRef | BuiltinType | EnumType
    init: Expression

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> Variable:
        var = node.field("var").text
        type = TypeRef.make(node.field("type"), scope)
        init = parse_expression(node.field("init"), scope)
        obj = cls(var, type, init)
        scope.define(var, obj, node.pos)
        return obj

    def resolve(self):
        if isinstance(self.type, TypeRef):
            self.type = self.type.resolve_builtin()
        self.init = resolve_expression(self.init)


@attrs.define
class NodeField:
    name: str
    type: TypeRef | BuiltinType | EnumType
    is_node_key: bool
    is_static: bool
    pos: SourcePosition | None = attrs.field(default=None, repr=False, eq=False)

    @classmethod
    def make(cls, node: PTNode, scope: Scope):
        name = node.field("name").text
        type = TypeRef.make(node.field("type"), scope)

        annotations = [c.text for c in node.fields("annotation")]
        is_node_key = "node key" in annotations
        is_static = "static" in annotations or is_node_key

        obj = NodeField(name, type, is_node_key, is_static, node.pos)
        scope.define(name, obj, node.pos)
        return obj

    def resolve(self):
        if isinstance(self.type, TypeRef):
            self.type = self.type.resolve()


@attrs.define
class NodeTable:
    fields: list[NodeField]
    key: NodeField
    contagions: list[Contagion]
    scope: Scope | None = attrs.field(default=None, repr=False, eq=False)
    pos: SourcePosition | None = attrs.field(default=None, repr=False, eq=False)

    @classmethod
    def make(cls, node: PTNode, parent_scope: Scope) -> NodeTable:
        scope = Scope(name="node_table", parent=parent_scope)

        fields = []
        for child in node.named_children:
            fields.append(NodeField.make(child, scope))

        key = [f for f in fields if f.is_node_key]
        if len(key) != 1:
            raise Error(
                "Node key error",
                "One and only one node key field must be specified.",
                node.pos,
            )
        key = key[0]

        obj = NodeTable(fields, key, [], scope, node.pos)
        parent_scope.define("node_table", obj, node.pos)
        return obj

    def add_contagions(self, contagions: list[Contagion]):
        assert self.scope is not None

        for contagion in contagions:
            self.scope.define(contagion.name, contagion, contagion.pos)
            self.contagions.append(contagion)

    def resolve(self):
        for field in self.fields:
            field.resolve()


@attrs.define
class EdgeField:
    name: str
    type: TypeRef | BuiltinType | EnumType
    is_target_node_key: bool
    is_source_node_key: bool
    is_static: bool
    pos: SourcePosition | None = attrs.field(default=None, repr=False, eq=False)

    @classmethod
    def make(cls, node: PTNode, scope: Scope):
        name = node.field("name").text
        type = TypeRef.make(node.field("type"), scope)

        annotations = [c.text for c in node.fields("annotation")]
        is_source_node_key = "source node key" in annotations
        is_target_node_key = "target node key" in annotations
        is_static = "static" in annotations or is_source_node_key or is_target_node_key

        obj = EdgeField(
            name, type, is_target_node_key, is_source_node_key, is_static, node.pos
        )
        scope.define(name, obj, node.pos)
        return obj

    def resolve(self):
        if isinstance(self.type, TypeRef):
            self.type = self.type.resolve()


@attrs.define
class TargetNodeAccessor:
    scope: Scope | None = attrs.field(default=None, repr=False, eq=False)


@attrs.define
class SourceNodeAccessor:
    scope: Scope | None = attrs.field(default=None, repr=False, eq=False)


@attrs.define
class EdgeTable:
    fields: list[EdgeField]
    target_node_key: EdgeField
    source_node_key: EdgeField
    contagions: list[Contagion]
    source_node: SourceNodeAccessor
    target_node: TargetNodeAccessor
    scope: Scope | None = attrs.field(default=None, repr=False, eq=False)
    pos: SourcePosition | None = attrs.field(default=None, repr=False, eq=False)

    @classmethod
    def make(cls, node: PTNode, parent_scope: Scope) -> EdgeTable:
        scope = Scope(name="edge_table", parent=parent_scope)

        fields = []
        for child in node.named_children:
            fields.append(EdgeField.make(child, scope))

        target_key = [f for f in fields if f.is_target_node_key]
        if len(target_key) != 1:
            raise Error(
                "Edge target key error",
                "One and only one target node key field must be specified.",
                node.pos,
            )
        target_key = target_key[0]

        source_key = [f for f in fields if f.is_source_node_key]
        if len(source_key) != 1:
            raise Error(
                "Edge source key error",
                "One and only one source node key field must be specified.",
                node.pos,
            )
        source_key = source_key[0]

        source_node = SourceNodeAccessor(None)
        scope.define("source_node", source_node, None)

        target_node = TargetNodeAccessor(None)
        scope.define("target_node", target_node, None)

        obj = EdgeTable(
            fields,
            target_key,
            source_key,
            [],
            source_node,
            target_node,
            scope,
            node.pos,
        )
        parent_scope.define("edge_table", obj, node.pos)
        return obj

    def add_contagions(self, contagions: list[Contagion]):
        assert self.scope is not None

        for contagion in contagions:
            self.scope.define(contagion.name, contagion, contagion.pos)
            self.contagions.append(contagion)

    def resolve(self):
        assert self.scope is not None

        for field in self.fields:
            field.resolve()

        node_table = self.scope.resolve("node_table", None)
        self.source_node.scope = node_table.scope
        self.target_node.scope = node_table.scope


@attrs.define
class ConstantDist:
    name: str
    v: int | float
    pos: SourcePosition | None = attrs.field(default=None, repr=False, eq=False)

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> ConstantDist:
        name = node.field("name").text
        v = make_literal(node.field("v"), int | float)
        obj = ConstantDist(name, v, node.pos)
        scope.define(name, obj, node.pos)
        return obj


@attrs.define
class DiscreteDist:
    name: str
    ps: list[int | float]
    vs: list[int | float]
    pos: SourcePosition | None = attrs.field(default=None, repr=False, eq=False)

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> DiscreteDist:
        name = node.field("name").text
        ps = []
        vs = []
        for child in node.fields("pv"):
            p = make_literal(child.field("p"), int | float)
            ps.append(p)

            v = make_literal(child.field("v"), int | float)
            vs.append(v)

        obj = DiscreteDist(name, ps, vs, node.pos)
        scope.define(name, obj, node.pos)
        return obj


@attrs.define
class NormalDist:
    name: str
    mean: int | float
    std: int | float
    min: int | float
    max: int | float
    pos: SourcePosition | None = attrs.field(default=None, repr=False, eq=False)

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> NormalDist:
        name = node.field("name").text
        mean = make_literal(node.field("mean"), int | float)
        std = make_literal(node.field("std"), int | float)

        min = node.maybe_field("min")
        min = -1e300 if min is None else make_literal(min, int | float)

        max = node.maybe_field("max")
        max = 1e300 if max is None else make_literal(max, int | float)

        obj = NormalDist(name, mean, std, min, max, node.pos)
        scope.define(name, obj, node.pos)
        return obj


@attrs.define
class UniformDist:
    name: str
    low: int | float
    high: int | float
    pos: SourcePosition | None = attrs.field(default=None, repr=False, eq=False)

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> UniformDist:
        name = node.field("name").text
        low = make_literal(node.field("low"), int | float)
        high = make_literal(node.field("high"), int | float)
        obj = UniformDist(name, low, high, node.pos)
        scope.define(name, obj, node.pos)
        return obj


Distribution = ConstantDist | DiscreteDist | NormalDist | UniformDist


def parse_distribution(node: PTNode, scope: Scope) -> Distribution:
    match node.type:
        case "constant_dist":
            return ConstantDist.make(node, scope)
        case "discrete_dist":
            return DiscreteDist.make(node, scope)
        case "normal_dist":
            return NormalDist.make(node, scope)
        case "uniform_dist":
            return UniformDist.make(node, scope)
        case unexpected:
            raise UnexpectedValue(unexpected, node.pos)


@attrs.define
class DistRef:
    const: str
    scope: Scope | None = attrs.field(default=None, repr=False, eq=False)
    pos: SourcePosition | None = attrs.field(default=None, repr=False, eq=False)

    def __str__(self):
        return self.const

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> DistRef:
        type = node.text
        return DistRef(type, scope, node.pos)

    def resolve(self) -> Distribution:
        assert self.scope is not None
        v = self.scope.resolve(self.const, self.pos)

        if isinstance(v, Distribution):
            return v

        raise Error(
            "Invalid value",
            "Expected a reference to a distribution",
            self.pos,
        )


@attrs.define
class Transition:
    entry: ConstRef | EnumConstant
    exit: ConstRef | EnumConstant
    p: int | float
    dwell: DistRef | Distribution
    pos: SourcePosition | None = attrs.field(default=None, repr=False, eq=False)

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> Transition:
        entry = ConstRef.make(node.field("entry"), scope)
        exit = ConstRef.make(node.field("exit"), scope)
        p_node = node.maybe_field("p")
        if p_node is None:
            p = 1.0
        else:
            p = make_literal(p_node, int | float)
        dwell = DistRef.make(node.field("dwell"), scope)
        return cls(entry, exit, p, dwell, node.pos)

    def resolve(self):
        if isinstance(self.entry, ConstRef):
            self.entry = self.entry.resolve()
        if isinstance(self.exit, ConstRef):
            self.exit = self.exit.resolve()
        if isinstance(self.dwell, DistRef):
            self.dwell = self.dwell.resolve()


@attrs.define
class Transmission:
    contact: ConstRef | EnumConstant
    entry: ConstRef | EnumConstant
    exit: ConstRef | EnumConstant
    pos: SourcePosition | None = attrs.field(default=None, repr=False, eq=False)

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> Transmission:
        contact = ConstRef.make(node.field("contact"), scope)
        entry = ConstRef.make(node.field("entry"), scope)
        exit = ConstRef.make(node.field("exit"), scope)
        return cls(contact, entry, exit, node.pos)

    def resolve(self):
        if isinstance(self.contact, ConstRef):
            self.contact = self.contact.resolve()
        if isinstance(self.entry, ConstRef):
            self.entry = self.entry.resolve()
        if isinstance(self.exit, ConstRef):
            self.exit = self.exit.resolve()


@attrs.define
class Contagion:
    name: str
    state_type: TypeRef | EnumType
    transitions: list[Transition]
    transmissions: list[Transmission]
    susceptibility: Function
    infectivity: Function
    transmissibility: Function
    enabled: Function
    node_fields: list[NodeField]
    scope: Scope | None = attrs.field(default=None, repr=False, eq=False)
    pos: SourcePosition | None = attrs.field(default=None, repr=False, eq=False)

    @classmethod
    def make(cls, node: PTNode, parent_scope: Scope) -> Contagion:
        name = node.field("name").text
        scope = Scope(name=name, parent=parent_scope)

        state_type = None
        transitions = []
        transmissions = []
        susceptibility = None
        infectivity = None
        transmissibility = None
        enabled = None

        for child in node.fields("body"):
            match child.type:
                case "contagion_state_type":
                    if state_type is not None:
                        raise Error(
                            "Multiple state types",
                            "Contagion state type is multiply defined",
                            child.pos,
                        )
                    state_type = TypeRef.make(child.field("type"), scope)
                case "transitions":
                    for gc in child.fields("body"):
                        transitions.append(Transition.make(gc, scope))
                case "transmissions":
                    for gc in child.fields("body"):
                        transmissions.append(Transmission.make(gc, scope))
                case "function":
                    match child.field("name").text:
                        case "susceptibility":
                            if susceptibility is not None:
                                raise Error(
                                    "Multiple susceptibility",
                                    "Susceptibility function is multiply defined",
                                    child.pos,
                                )
                            susceptibility = Function.make(child, scope)
                        case "infectivity":
                            if infectivity is not None:
                                raise Error(
                                    "Multiple infectivity",
                                    "Infectivity function is multiply defined",
                                    child.pos,
                                )
                            infectivity = Function.make(child, scope)
                        case "transmissibility":
                            if transmissibility is not None:
                                raise Error(
                                    "Multiple transmissibility",
                                    "Transmissibility function is multiply defined",
                                    child.pos,
                                )
                            transmissibility = Function.make(child, scope)
                        case "enabled":
                            if enabled is not None:
                                raise Error(
                                    "Multiple enabled",
                                    "Enabled (edge) function is multiply defined",
                                    child.pos,
                                )
                            enabled = Function.make(child, scope)
                        case _:
                            raise Error(
                                "Invalid function",
                                f"{child.field('name').text} is not a valid contagion function",
                                child.field("type").pos,
                            )
                case _ as unexpcted:
                    raise UnexpectedValue(unexpcted, child.pos)

        if state_type is None:
            raise Error(
                "State type undefined", "State type has not been defined", node.pos
            )
        if susceptibility is None:
            raise Error(
                "Susceptibility undefined",
                "Susceptibility function has not been defined",
                node.pos,
            )
        if infectivity is None:
            raise Error(
                "Infectivity undefined",
                "Infectivity function has not been defined",
                node.pos,
            )
        if transmissibility is None:
            raise Error(
                "Transmissibility undefined",
                "Transmissibility function has not been defined",
                node.pos,
            )
        if enabled is None:
            raise Error(
                "Edge enabled undefined",
                "Enabled (edge) function has not been defined",
                node.pos,
            )

        node_fields = [
            NodeField("state", state_type, False, False, None),
            NodeField("next_state", state_type, False, False, None),
            NodeField("dwell_time", BuiltinType("float"), False, False, None),
        ]
        for field in node_fields:
            scope.define(field.name, field, field.pos)

        obj = Contagion(
            name,
            state_type,
            transitions,
            transmissions,
            susceptibility,
            infectivity,
            transmissibility,
            enabled,
            node_fields,
            scope,
            node.pos,
        )
        parent_scope.define(name, obj, node.pos)
        return obj

    def resolve(self):
        if isinstance(self.state_type, TypeRef):
            self.state_type = self.state_type.resolve_enum()

        for field in self.node_fields:
            field.resolve()

        for transm in self.transmissions:
            transm.resolve()

        for transi in self.transitions:
            transi.resolve()

        self.susceptibility.resolve()
        self.infectivity.resolve()
        self.transmissibility.resolve()
        self.enabled.resolve()

        self.susceptibility.check_susceptibility()
        self.infectivity.check_infectivity()
        self.transmissibility.check_transmissibility()
        self.enabled.check_enabled()


@attrs.define
class PassStatement:
    pos: SourcePosition | None = attrs.field(default=None, repr=False, eq=False)

    def resolve(self):
        pass


@attrs.define
class ReturnStatement:
    expression: Expression
    pos: SourcePosition | None = attrs.field(default=None, repr=False, eq=False)

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> ReturnStatement:
        expression = node.named_children[0]
        expression = parse_expression(expression, scope)
        return ReturnStatement(expression, node.pos)

    def resolve(self):
        self.expression = resolve_expression(self.expression)


@attrs.define
class ElifSection:
    condition: Expression
    body: list[Statement]

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> ElifSection:
        condition = parse_expression(node.field("condition"), scope)
        body = []
        for child in node.fields("body"):
            body.append(parse_statement(child, scope))
        return cls(condition, body)

    def resolve(self):
        self.condition = resolve_expression(self.condition)
        for stmt in self.body:
            stmt.resolve()


@attrs.define
class ElseSection:
    body: list[Statement]

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> ElseSection:
        body = []
        for child in node.fields("body"):
            body.append(parse_statement(child, scope))
        return cls(body)

    def resolve(self):
        for stmt in self.body:
            stmt.resolve()


@attrs.define
class IfStatement:
    condition: Expression
    body: list[Statement]
    elifs: list[ElifSection]
    else_: ElseSection | None

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> IfStatement:
        condition = parse_expression(node.field("condition"), scope)
        body = []
        for child in node.fields("body"):
            body.append(parse_statement(child, scope))
        elifs = []
        for child in node.fields("elif"):
            elifs.append(ElifSection.make(child, scope))
        else_ = node.maybe_field("else")
        if else_ is not None:
            else_ = ElseSection.make(else_, scope)

        obj = cls(condition, body, elifs, else_)
        return obj

    def resolve(self):
        self.condition = resolve_expression(self.condition)
        for stmt in self.body:
            stmt.resolve()
        for elif_ in self.elifs:
            elif_.resolve()
        if self.else_ is not None:
            self.else_.resolve()


@attrs.define
class WhileLoop:
    condition: Expression
    body: list[Statement]

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> WhileLoop:
        condition = parse_expression(node.field("condition"), scope)
        body = []
        for child in node.fields("body"):
            body.append(parse_statement(child, scope))
        return cls(condition, body)

    def resolve(self):
        self.condition = resolve_expression(self.condition)
        for stmt in self.body:
            stmt.resolve()


@attrs.define
class CallStatement:
    call: FunctionCall

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> CallStatement:
        call = FunctionCall.make(node.named_children[0], scope)
        return cls(call)

    def resolve(self):
        self.call.resolve()


@attrs.define
class UpdateStatement:
    left: Reference
    operator: str
    right: Expression

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> UpdateStatement:
        left = Reference.make(node.field("left"), scope)
        operator = node.field("operator").text
        right = parse_expression(node.field("right"), scope)
        obj = cls(left, operator, right)
        return obj

    def resolve(self):
        self.left.resolve()
        self.right = resolve_expression(self.right)


@attrs.define
class PrintStatement:
    args: list[str | Expression]
    sep: str
    end: str
    pos: SourcePosition | None = attrs.field(default=None, repr=False, eq=False)

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> PrintStatement:
        args = []
        for child in node.named_children:
            if child.type == "string":
                args.append(child.text)
            else:
                args.append(parse_expression(child, scope))
        sep = '" "'
        end = r'"\n"'
        return PrintStatement(args, sep, end, node.pos)

    def resolve(self):
        args = []
        for arg in self.args:
            if isinstance(arg, str):
                args.append(arg)
            else:
                args.append(resolve_expression(arg))
        self.args = args


@attrs.define
class UnaryExpression:
    operator: str
    argument: Expression

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> UnaryExpression:
        operator = node.field("operator").text
        argument = parse_expression(node.field("argument"), scope)
        return cls(operator, argument)

    def resolve(self):
        self.argument = resolve_expression(self.argument)


@attrs.define
class BinaryExpression:
    left: Expression
    operator: str
    right: Expression

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> BinaryExpression:
        left = parse_expression(node.field("left"), scope)
        operator = node.field("operator").text
        right = parse_expression(node.field("right"), scope)
        return cls(left, operator, right)

    def resolve(self):
        self.left = resolve_expression(self.left)
        self.right = resolve_expression(self.right)


@attrs.define
class ParenthesizedExpression:
    expression: Expression

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> ParenthesizedExpression:
        expression = parse_expression(node.field("expression"), scope)
        return cls(expression)

    def resolve(self):
        self.expression = resolve_expression(self.expression)


@attrs.define
class Reference:
    names: list[str]
    refs: list
    scope: Scope | None = attrs.field(default=None, repr=False, eq=False)
    pos: SourcePosition | None = attrs.field(default=None, repr=False, eq=False)

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> Reference:
        names = [s.text for s in node.named_children]
        return Reference(names, [], scope, node.pos)

    def resolve(self):
        if not self.refs:
            assert self.scope is not None
            head, tail = self.names[0], self.names[1:]

            v = self.scope.resolve(head, self.pos)
            self.refs.append(v)

            for part in tail:
                v = v.scope.resolve(part, self.pos)
                self.refs.append(v)


@attrs.define
class FunctionCall:
    function: Reference
    args: list[Expression]

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> FunctionCall:
        function = Reference.make(node.field("function"), scope)
        args = [parse_expression(c, scope) for c in node.fields("args")]
        return cls(function, args)

    def resolve(self) -> Any:
        self.function.resolve()
        self.args = [resolve_expression(arg) for arg in self.args]


@attrs.define
class Param:
    name: str
    type: TypeRef | BuiltinType | EnumType
    scope: Scope | None = attrs.field(default=None, repr=False, eq=False)
    pos: SourcePosition | None = attrs.field(default=None, repr=False, eq=False)

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> Param:
        name = node.field("name").text
        type = TypeRef.make(node.field("type"), scope)
        if type.type == "node" or type.type == "edge":
            obj = cls(name, type, scope, node.pos)
        else:
            obj = cls(name, type, None, node.pos)
        scope.define(name, obj, node.pos)
        return obj

    def resolve(self):
        if isinstance(self.type, TypeRef):
            self.type = self.type.resolve()

        match self.type:
            case BuiltinType("node"):
                assert self.scope is not None
                node_table = self.scope.resolve("node_table", self.pos)
                self.scope = node_table.scope
            case BuiltinType("edge"):
                assert self.scope is not None
                edge_table = self.scope.resolve("edge_table", self.pos)
                self.scope = edge_table.scope


@attrs.define
class Function:
    name: str
    params: list[Param]
    variables: list[Variable]
    return_: TypeRef | BuiltinType | EnumType | None
    body: list[Variable | Statement]
    scope: Scope | None = attrs.field(default=None, repr=False, eq=False)
    pos: SourcePosition | None = attrs.field(default=None, repr=False, eq=False)

    @classmethod
    def make(cls, node: PTNode, parent_scope: Scope) -> Function:
        name = node.field("name").text
        scope = Scope(name=name, parent=parent_scope)

        params = []
        for child in node.field("params").named_children:
            params.append(Param.make(child, scope))

        return_node = node.maybe_field("return")
        if return_node is None:
            return_ = None
        else:
            return_ = TypeRef.make(return_node, scope)

        variables = []
        body = []
        for child in node.field("body").named_children:
            if child.type == "variable":
                var = Variable.make(child, scope)
                variables.append(var)
                body.append(var)
            else:
                body.append(parse_statement(child, scope))

        obj = Function(name, params, variables, return_, body, scope, node.pos)
        parent_scope.define(name, obj, node.pos)
        return obj

    def check_main(self):
        match self.params:
            case []:
                pass
            case _:
                raise Error(
                    "Bad main", "Main function must not have any parameters", self.pos
                )
        match self.return_:
            case None:
                pass
            case _:
                raise Error(
                    "Bad main", "Main function must not have a return type", self.pos
                )

    def check_susceptibility(self):
        match self.params:
            case [Param(_, BuiltinType("node"))]:
                pass
            case _:
                raise Error(
                    "Bad susceptibility",
                    "Susceptibility function must have one parameter of type `node'",
                    self.pos,
                )
        match self.return_:
            case BuiltinType("float"):
                pass
            case _:
                raise Error(
                    "Bad susceptibility",
                    "Susceptibility function must have a return type `float'",
                    self.pos,
                )

    def check_infectivity(self):
        match self.params:
            case [Param(_, BuiltinType("node"))]:
                pass
            case _:
                raise Error(
                    "Bad infectivity",
                    "Infectivity function must have one parameter of type `node'",
                    self.pos,
                )

        match self.return_:
            case BuiltinType("float"):
                pass
            case _:
                raise Error(
                    "Bad infectivity",
                    "Infectivity function must have a return type `float'",
                    self.pos,
                )

    def check_transmissibility(self):
        match self.params:
            case [Param(_, BuiltinType("edge"))]:
                pass
            case _:
                raise Error(
                    "Bad transmissibility",
                    "Transmissibility function must have one parameter of type `edge'",
                    self.pos,
                )

        match self.return_:
            case BuiltinType("float"):
                pass
            case _:
                raise Error(
                    "Bad transmissibility",
                    "Transmissibility function must have a return type `float'",
                    self.pos,
                )

    def check_enabled(self):
        match self.params:
            case [Param(_, BuiltinType("edge"))]:
                pass
            case _:
                raise Error(
                    "Bad enabled (edge)",
                    "Enabled (edge) function must have one parameter of type `edge'",
                    self.pos,
                )
        match self.return_:
            case BuiltinType("bool"):
                pass
            case _:
                raise Error(
                    "Bad enabled (edge)",
                    "Enabled (edge) function must have a return type `bool'",
                    self.pos,
                )

    def resolve(self):
        for param in self.params:
            param.resolve()

        match self.return_:
            case TypeRef():
                self.return_ = self.return_.resolve()

        for stmt in self.body:
            stmt.resolve()


Expression = (
    int
    | float
    | bool
    | UnaryExpression
    | BinaryExpression
    | ParenthesizedExpression
    | Reference
    | FunctionCall
)


def parse_expression(node: PTNode, scope: Scope) -> Expression:
    match node.type:
        case "integer" | "float" | "boolean":
            return make_literal(node, int | float | bool)
        case "unary_expression":
            return UnaryExpression.make(node, scope)
        case "binary_expression":
            return BinaryExpression.make(node, scope)
        case "parenthesized_expression":
            return ParenthesizedExpression.make(node, scope)
        case "reference":
            return Reference.make(node, scope)
        case "function_call":
            return FunctionCall.make(node, scope)
        case unexpected:
            raise UnexpectedValue(unexpected, node.pos)


def resolve_expression(e: Expression) -> Expression:
    match e:
        case UnaryExpression() | BinaryExpression() | ParenthesizedExpression():
            e.resolve()
            return e
        case Reference():
            e.resolve()
            return e
        case FunctionCall():
            e.resolve()
            return e
        case _:
            return e


Statement = (
    PassStatement
    | ReturnStatement
    | IfStatement
    | WhileLoop
    | CallStatement
    | UpdateStatement
    | PrintStatement
)


def parse_statement(node: PTNode, scope: Scope) -> Statement:
    match node.type:
        case "pass_statement":
            return PassStatement(node.pos)
        case "return_statement":
            return ReturnStatement.make(node, scope)
        case "if_statement":
            return IfStatement.make(node, scope)
        case "while_loop":
            return WhileLoop.make(node, scope)
        case "call_statement":
            return CallStatement.make(node, scope)
        case "update_statement":
            return UpdateStatement.make(node, scope)
        case "print_statement":
            return PrintStatement.make(node, scope)
        case unexpected:
            raise UnexpectedValue(unexpected, node.pos)


def add_builtins(scope: Scope, pos: SourcePosition):
    for type in ["int", "uint", "float", "bool", "node", "edge"]:
        add_builtin_type(type, scope, pos)
    for type in ["u8", "u16", "u32", "u64"]:
        add_builtin_type(type, scope, pos)
    for type in ["i8", "i16", "i32", "i64"]:
        add_builtin_type(type, scope, pos)
    for type in ["f32", "f64"]:
        add_builtin_type(type, scope, pos)


@attrs.define
class Source:
    module: str
    configs: list[Config]
    variables: list[Variable]
    enums: list[EnumType]
    node_table: NodeTable
    edge_table: EdgeTable
    distributions: list[Distribution]
    contagions: list[Contagion]
    functions: list[Function]
    main_function: Function
    scope: Scope | None = attrs.field(default=None, repr=False, eq=False)
    node: PTNode | None = attrs.field(default=None, repr=False, eq=False)

    @classmethod
    def make(cls, module: str, node: PTNode) -> Source:
        scope = Scope(name="source")
        add_builtins(scope, node.pos)

        configs: list[Config] = []
        variables: list[Variable] = []
        enums: list[EnumType] = []
        node_table: NodeTable | None = None
        edge_table: EdgeTable | None = None
        distributions: list[Distribution] = []
        contagions: list[Contagion] = []
        functions: list[Function] = []
        main_function: Function | None = None

        for child in node.named_children:
            match child.type:
                case "config":
                    configs.append(Config.make(child, scope))
                case "variable":
                    variables.append(Variable.make(child, scope))
                case "enum":
                    enums.append(EnumType.make(child, scope))
                case "node":
                    if node_table is not None:
                        raise Error(
                            "Multiple node tables",
                            "Node table is multiply defined",
                            child.pos,
                        )

                    node_table = NodeTable.make(child, scope)
                case "edge":
                    if edge_table is not None:
                        raise Error(
                            "Multiple edge tables",
                            "Edge table is multiply defined",
                            child.pos,
                        )

                    edge_table = EdgeTable.make(child, scope)
                case "distributions":
                    for d in child.named_children:
                        distributions.append(parse_distribution(d, scope))
                case "contagion":
                    contagions.append(Contagion.make(child, scope))
                case "function":
                    func = Function.make(child, scope)
                    if func.name == "main":
                        if main_function is not None:
                            raise Error(
                                "Multiple mains",
                                "Main function is multiply defined",
                                child.pos,
                            )

                        main_function = func
                    else:
                        functions.append(func)

        if node_table is None:
            raise Error("Node undefined", "Node table was not defined", node.pos)
        if edge_table is None:
            raise Error("Edge undefined", "Edge table was not defined", node.pos)

        if main_function is None:
            raise Error("Main undefined", "No main function was defined", node.pos)

        node_table.add_contagions(contagions)
        edge_table.add_contagions(contagions)

        return cls(
            module=module,
            configs=configs,
            variables=variables,
            enums=enums,
            node_table=node_table,
            edge_table=edge_table,
            distributions=distributions,
            contagions=contagions,
            functions=functions,
            main_function=main_function,
            scope=scope,
        )

    def resolve(self):
        for config in self.configs:
            config.resolve()
        for var in self.variables:
            var.resolve()
        self.node_table.resolve()
        self.edge_table.resolve()
        for contagion in self.contagions:
            contagion.resolve()
        for function in self.functions:
            function.resolve()
        self.main_function.resolve()

        self.main_function.check_main()


def mk_ast1(filename: Path, node: PTNode) -> Source:
    """Make ast1 form parse tree."""
    module = filename.stem.replace(".", "_")
    source = Source.make(module, node)
    return source


@click.command()
@click.argument(
    "filename",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
)
def print_ast1(filename: Path):
    """Print the AST1."""
    file_bytes = filename.read_bytes()
    try:
        pt = mk_pt(str(filename), file_bytes)
        ast1 = mk_ast1(filename, pt)
        rich.print(ast1)
    except (ParseTreeConstructionError, ASTConstructionError) as e:
        e.rich_print()


@click.command()
@click.argument(
    "filename",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
)
def print_ast2(filename: Path):
    """Print the AST2."""
    file_bytes = filename.read_bytes()
    try:
        pt = mk_pt(str(filename), file_bytes)
        ast1 = mk_ast1(filename, pt)
        ast1.resolve()
        rich.print(ast1)
    except (ParseTreeConstructionError, ASTConstructionError) as e:
        e.rich_print()
