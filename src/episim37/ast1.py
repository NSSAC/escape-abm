"""Abstract syntax tree v1."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeVar, Generic

import attrs
import click
import rich
import rich.markup
from typeguard import check_type

from .parse_tree import mk_pt, PTNode, ParseTreeConstructionError, SourcePosition


T = TypeVar("T")


class UniqueObject(Generic[T]):
    def __init__(self):
        self.obj: T | None = None

    def dup_error(self, type: str, explanation: str, pos: SourcePosition):
        if self.obj is not None:
            raise Error(type, explanation, pos)

    def set(self, obj: T):
        assert self.obj is None
        self.obj = obj

    def get(self) -> T:
        assert self.obj is not None
        return self.obj

    def missing_error(self, type: str, explanation, pos: SourcePosition):
        if self.obj is None:
            raise Error(type, explanation, pos)


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
        case _ as unexpected:
            raise UnexpectedValue(unexpected, n.pos)

    if not isinstance(v, types):
        raise UnexpectedValue(v, n.pos)

    return v


@attrs.define
class BuiltinType:
    name: str


@attrs.define
class BuiltinTypeRef:
    type: str
    scope: Scope | None = attrs.field(default=None, repr=False, eq=False)
    pos: SourcePosition | None = attrs.field(default=None, repr=False, eq=False)

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> BuiltinTypeRef:
        return cls(node.text, scope, node.pos)

    def resolve(self) -> BuiltinType:
        assert self.scope is not None

        v = self.scope.resolve(self.type, self.pos)
        if isinstance(v, BuiltinType):
            return v

        raise Error(
            "Invalid type",
            "Expected a built type",
            self.pos,
        )


@attrs.define
class BuiltinFunction:
    name: str
    params: list[Param]
    return_: TypeRef | None


@attrs.define
class EnumConstant:
    name: str
    type: EnumType


@attrs.define
class EnumConstantRef:
    const: str
    scope: Scope | None = attrs.field(default=None, repr=False, eq=False)
    pos: SourcePosition | None = attrs.field(default=None, repr=False, eq=False)

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> EnumConstantRef:
        return cls(node.text, scope, node.pos)

    def resolve(self) -> EnumConstant:
        assert self.scope is not None

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
    consts: list[str]
    pos: SourcePosition | None = attrs.field(default=None, repr=False, eq=False)

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> EnumType:
        name = node.field("name").text
        consts = [child.text for child in node.fields("const")]

        obj = cls(name, consts, node.pos)
        for child in node.fields("const"):
            enum_const = EnumConstant(child.text, obj)
            scope.define(enum_const.name, enum_const, child.pos)

        scope.define(name, obj, node.pos)
        return obj


@attrs.define
class EnumTypeRef:
    type: str
    scope: Scope | None = attrs.field(default=None, repr=False, eq=False)
    pos: SourcePosition | None = attrs.field(default=None, repr=False, eq=False)

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> EnumTypeRef:
        return cls(node.text, scope, node.pos)

    def resolve(self) -> EnumType:
        assert self.scope is not None

        v = self.scope.resolve(self.type, self.pos)
        if isinstance(v, EnumType):
            return v

        raise Error(
            "Invalid type",
            "Expected an enumerated type",
            self.pos,
        )


@attrs.define
class TypeRef:
    type: str
    scope: Scope | None = attrs.field(default=None, repr=False, eq=False)
    pos: SourcePosition | None = attrs.field(default=None, repr=False, eq=False)

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> TypeRef:
        return cls(node.text, scope, node.pos)

    def resolve(self) -> BuiltinType | EnumType:
        assert self.scope is not None

        v = self.scope.resolve(self.type, self.pos)
        if isinstance(v, BuiltinType | EnumType):
            return v

        raise Error(
            "Invalid type",
            "Expected a built in or enumerated type",
            self.pos,
        )


@attrs.define
class Config:
    name: str
    type: BuiltinTypeRef
    default: int | float | bool
    pos: SourcePosition | None = attrs.field(default=None, repr=False, eq=False)

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> Config:
        name = node.field("name").text
        type = BuiltinTypeRef.make(node.field("type"), scope)
        default = make_literal(node.field("default"), int | float | bool)
        obj = cls(name, type, default, node.pos)
        scope.define(name, obj, node.pos)
        return obj


@attrs.define
class Variable:
    var: str
    type: TypeRef
    init: Expression
    scope: Scope | None = attrs.field(default=None, repr=False, eq=False)
    pos: SourcePosition | None = attrs.field(default=None, repr=False, eq=False)

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> Variable:
        var = node.field("var").text
        type = TypeRef.make(node.field("type"), scope)
        init = parse_expression(node.field("init"), scope)
        obj = cls(var, type, init, None, node.pos)
        scope.define(var, obj, node.pos)
        return obj

    def link_tables(self, node_table: NodeTable, edge_table: EdgeTable):
        match self.type:
            case TypeRef("node"):
                self.scope = node_table.scope
            case TypeRef("edge"):
                self.scope = edge_table.scope


@attrs.define
class NodeField:
    name: str
    type: TypeRef
    is_node_key: bool
    is_static: bool
    pos: SourcePosition | None = attrs.field(default=None, repr=False, eq=False)

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> NodeField:
        name = node.field("name").text
        type = TypeRef.make(node.field("type"), scope)

        annotations = [c.text for c in node.fields("annotation")]
        is_node_key = "node key" in annotations
        is_static = "static" in annotations or is_node_key

        obj = cls(name, type, is_node_key, is_static, node.pos)
        scope.define(name, obj, node.pos)
        return obj


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

        obj = cls(fields, key, [], scope, node.pos)
        parent_scope.define("node_table", obj, node.pos)
        return obj

    def link_contagions(self, contagions: list[Contagion]):
        assert self.scope is not None

        for contagion in contagions:
            self.scope.define(contagion.name, contagion, contagion.pos)
            self.contagions.append(contagion)


@attrs.define
class EdgeField:
    name: str
    type: TypeRef
    is_target_node_key: bool
    is_source_node_key: bool
    is_static: bool
    pos: SourcePosition | None = attrs.field(default=None, repr=False, eq=False)

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> EdgeField:
        name = node.field("name").text
        type = TypeRef.make(node.field("type"), scope)

        annotations = [c.text for c in node.fields("annotation")]
        is_source_node_key = "source node key" in annotations
        is_target_node_key = "target node key" in annotations
        is_static = "static" in annotations or is_source_node_key or is_target_node_key

        obj = cls(
            name, type, is_target_node_key, is_source_node_key, is_static, node.pos
        )
        scope.define(name, obj, node.pos)
        return obj


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

        obj = cls(
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

    def link_contagions(self, contagions: list[Contagion]):
        assert self.scope is not None

        for contagion in contagions:
            self.scope.define(contagion.name, contagion, contagion.pos)
            self.contagions.append(contagion)

    def link_node_table(self, node_table: NodeTable):
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
        obj = cls(name, v, node.pos)
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

        obj = cls(name, ps, vs, node.pos)
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

        obj = cls(name, mean, std, min, max, node.pos)
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
        obj = cls(name, low, high, node.pos)
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
    dist: str
    scope: Scope | None = attrs.field(default=None, repr=False, eq=False)
    pos: SourcePosition | None = attrs.field(default=None, repr=False, eq=False)

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> DistRef:
        return cls(node.text, scope, node.pos)

    def resolve(self) -> Distribution:
        assert self.scope is not None

        v = self.scope.resolve(self.dist, self.pos)
        if isinstance(v, Distribution):
            return v

        raise Error(
            "Invalid value",
            "Expected a reference to a distribution",
            self.pos,
        )


@attrs.define
class Transition:
    entry: EnumConstantRef
    exit: EnumConstantRef
    p: int | float
    dwell: DistRef
    pos: SourcePosition | None = attrs.field(default=None, repr=False, eq=False)

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> Transition:
        entry = EnumConstantRef.make(node.field("entry"), scope)
        exit = EnumConstantRef.make(node.field("exit"), scope)
        p_node = node.maybe_field("p")
        if p_node is None:
            p = 1.0
        else:
            p = make_literal(p_node, int | float)
        dwell = DistRef.make(node.field("dwell"), scope)
        return cls(entry, exit, p, dwell, node.pos)


@attrs.define
class Transmission:
    contact: EnumConstantRef
    entry: EnumConstantRef
    exit: EnumConstantRef
    pos: SourcePosition | None = attrs.field(default=None, repr=False, eq=False)

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> Transmission:
        contact = EnumConstantRef.make(node.field("contact"), scope)
        entry = EnumConstantRef.make(node.field("entry"), scope)
        exit = EnumConstantRef.make(node.field("exit"), scope)
        return cls(contact, entry, exit, node.pos)


@attrs.define
class Contagion:
    name: str
    state_type: EnumTypeRef
    transitions: list[Transition]
    transmissions: list[Transmission]
    susceptibility: Function
    infectivity: Function
    transmissibility: Function
    enabled: Function
    node_fields: list[NodeField]
    do_transmission: BuiltinFunction
    do_transition: BuiltinFunction
    scope: Scope | None = attrs.field(default=None, repr=False, eq=False)
    pos: SourcePosition | None = attrs.field(default=None, repr=False, eq=False)

    @classmethod
    def make(cls, node: PTNode, parent_scope: Scope) -> Contagion:
        name = node.field("name").text
        scope = Scope(name=name, parent=parent_scope)

        state_type: UniqueObject[EnumTypeRef] = UniqueObject()
        transitions: list[Transition] = []
        transmissions: list[Transmission] = []
        susceptibility: UniqueObject[Function] = UniqueObject()
        infectivity: UniqueObject[Function] = UniqueObject()
        transmissibility: UniqueObject[Function] = UniqueObject()
        enabled: UniqueObject[Function] = UniqueObject()

        for child in node.fields("body"):
            match child.type:
                case "contagion_state_type":
                    state_type.dup_error(
                        "Multiple state types",
                        "Contagion state type is multiply defined",
                        child.pos,
                    )
                    state_type.set(EnumTypeRef.make(child.field("type"), scope))
                case "transitions":
                    for gc in child.fields("body"):
                        transitions.append(Transition.make(gc, scope))
                case "transmissions":
                    for gc in child.fields("body"):
                        transmissions.append(Transmission.make(gc, scope))
                case "function":
                    match child.field("name").text:
                        case "susceptibility":
                            susceptibility.dup_error(
                                "Multiple susceptibility",
                                "Susceptibility function is multiply defined",
                                child.pos,
                            )
                            susceptibility.set(make_susceptibility(child, scope))
                        case "infectivity":
                            infectivity.dup_error(
                                "Multiple infectivity",
                                "Infectivity function is multiply defined",
                                child.pos,
                            )
                            infectivity.set(make_infectivity(child, scope))
                        case "transmissibility":
                            transmissibility.dup_error(
                                "Multiple transmissibility",
                                "Transmissibility function is multiply defined",
                                child.pos,
                            )
                            transmissibility.set(make_transmissibility(child, scope))
                        case "enabled":
                            enabled.dup_error(
                                "Multiple enabled",
                                "Enabled (edge) function is multiply defined",
                                child.pos,
                            )
                            enabled.set(make_enabled(child, scope))
                        case _:
                            raise Error(
                                "Invalid function",
                                f"{child.field('name').text} is not a valid contagion function",
                                child.field("type").pos,
                            )
                case _ as unexpected:
                    raise UnexpectedValue(unexpected, child.pos)

        state_type.missing_error(
            "State type undefined", "State type has not been defined", node.pos
        )
        susceptibility.missing_error(
            "Susceptibility undefined",
            "Susceptibility function has not been defined",
            node.pos,
        )
        infectivity.missing_error(
            "Infectivity undefined",
            "Infectivity function has not been defined",
            node.pos,
        )
        transmissibility.missing_error(
            "Transmissibility undefined",
            "Transmissibility function has not been defined",
            node.pos,
        )
        enabled.missing_error(
            "Edge enabled undefined",
            "Enabled (edge) function has not been defined",
            node.pos,
        )

        node_fields = [
            NodeField(
                "state",
                TypeRef(state_type.get().type, scope),
                False,
                False,
                None,
            ),
            NodeField(
                "next_state",
                TypeRef(state_type.get().type, scope),
                False,
                False,
                None,
            ),
            NodeField("dwell_time", TypeRef("float", scope), False, False, None),
        ]
        for field in node_fields:
            scope.define(field.name, field, field.pos)

        do_transmission = BuiltinFunction("do_transmission", [], None)
        do_transition = BuiltinFunction(
            "do_transition", [Param("elapsed", TypeRef("float", scope))], None
        )
        scope.define("do_transmission", do_transmission, None)
        scope.define("do_transition", do_transition, None)

        obj = cls(
            name,
            state_type.get(),
            transitions,
            transmissions,
            susceptibility.get(),
            infectivity.get(),
            transmissibility.get(),
            enabled.get(),
            node_fields,
            do_transmission,
            do_transition,
            scope,
            node.pos,
        )
        parent_scope.define(name, obj, node.pos)
        return obj


@attrs.define
class UnaryExpression:
    operator: str
    argument: Expression

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> UnaryExpression:
        operator = node.field("operator").text
        argument = parse_expression(node.field("argument"), scope)
        return cls(operator, argument)


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


@attrs.define
class ParenthesizedExpression:
    expression: Expression

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> ParenthesizedExpression:
        expression = parse_expression(node.field("expression"), scope)
        return cls(expression)


@attrs.define
class Reference:
    names: list[str]
    scope: Scope | None = attrs.field(default=None, repr=False, eq=False)
    pos: SourcePosition | None = attrs.field(default=None, repr=False, eq=False)

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> Reference:
        names = [s.text for s in node.named_children]
        return cls(names, scope, node.pos)

    def resolve(self) -> Referable:
        return resolve_ref(self)


@attrs.define
class FunctionCall:
    function: Reference
    args: list[Expression]

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> FunctionCall:
        function = Reference.make(node.field("function"), scope)
        args = [parse_expression(c, scope) for c in node.fields("args")]
        return cls(function, args)


@attrs.define
class PassStatement:
    pos: SourcePosition | None = attrs.field(default=None, repr=False, eq=False)


@attrs.define
class ReturnStatement:
    expression: Expression
    pos: SourcePosition | None = attrs.field(default=None, repr=False, eq=False)

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> ReturnStatement:
        expression = node.named_children[0]
        expression = parse_expression(expression, scope)
        return cls(expression, node.pos)


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


@attrs.define
class ElseSection:
    body: list[Statement]

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> ElseSection:
        body = []
        for child in node.fields("body"):
            body.append(parse_statement(child, scope))
        return cls(body)


@attrs.define
class IfStatement:
    condition: Expression
    body: list[Statement]
    elifs: list[ElifSection]
    else_: ElseSection | None
    pos: SourcePosition | None = attrs.field(default=None, repr=False, eq=False)

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

        obj = cls(condition, body, elifs, else_, node.pos)
        return obj


@attrs.define
class WhileLoop:
    condition: Expression
    body: list[Statement]
    pos: SourcePosition | None = attrs.field(default=None, repr=False, eq=False)

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> WhileLoop:
        condition = parse_expression(node.field("condition"), scope)
        body = []
        for child in node.fields("body"):
            body.append(parse_statement(child, scope))
        return cls(condition, body, node.pos)


@attrs.define
class CallStatement:
    call: FunctionCall
    pos: SourcePosition | None = attrs.field(default=None, repr=False, eq=False)

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> CallStatement:
        call = FunctionCall.make(node.named_children[0], scope)
        return cls(call, node.pos)


@attrs.define
class UpdateStatement:
    left: Reference
    operator: str
    right: Expression
    pos: SourcePosition | None = attrs.field(default=None, repr=False, eq=False)

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> UpdateStatement:
        left = Reference.make(node.field("left"), scope)
        operator = node.field("operator").text
        right = parse_expression(node.field("right"), scope)
        obj = cls(left, operator, right, node.pos)
        return obj


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
        return cls(args, sep, end, node.pos)


@attrs.define
class Param:
    name: str
    type: TypeRef
    scope: Scope | None = attrs.field(default=None, repr=False, eq=False)
    pos: SourcePosition | None = attrs.field(default=None, repr=False, eq=False)

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> Param:
        name = node.field("name").text
        type = TypeRef.make(node.field("type"), scope)
        obj = cls(name, type, None, node.pos)
        scope.define(name, obj, node.pos)
        return obj

    def link_tables(self, node_table: NodeTable, edge_table: EdgeTable):
        match self.type:
            case TypeRef("node"):
                self.scope = node_table.scope
            case TypeRef("edge"):
                self.scope = edge_table.scope


@attrs.define
class Function:
    name: str
    params: list[Param]
    variables: list[Variable]
    return_: TypeRef | None
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

        obj = cls(name, params, variables, return_, body, scope, node.pos)
        parent_scope.define(name, obj, node.pos)
        return obj

    def link_tables(self, node_table: NodeTable, edge_table: EdgeTable):
        for param in self.params:
            param.link_tables(node_table, edge_table)
        for var in self.variables:
            var.link_tables(node_table, edge_table)


def make_main(node: PTNode, parent_scope: Scope) -> Function:
    f = Function.make(node, parent_scope)
    match f.params:
        case []:
            pass
        case _:
            raise Error("Bad main", "Main function must not have any parameters", f.pos)
    match f.return_:
        case None:
            pass
        case _:
            raise Error("Bad main", "Main function must not have a return type", f.pos)
    return f


def make_susceptibility(node: PTNode, parent_scope: Scope) -> Function:
    f = Function.make(node, parent_scope)
    match f.params:
        case [Param(_, TypeRef("node"))]:
            pass
        case _:
            raise Error(
                "Bad susceptibility",
                "Susceptibility function must have one parameter of type `node'",
                f.pos,
            )
    match f.return_:
        case TypeRef("float"):
            pass
        case _:
            raise Error(
                "Bad susceptibility",
                "Susceptibility function must have a return type `float'",
                f.pos,
            )
    return f


def make_infectivity(node: PTNode, parent_scope: Scope) -> Function:
    f = Function.make(node, parent_scope)
    match f.params:
        case [Param(_, TypeRef("node"))]:
            pass
        case _:
            raise Error(
                "Bad infectivity",
                "Infectivity function must have one parameter of type `node'",
                f.pos,
            )

    match f.return_:
        case TypeRef("float"):
            pass
        case _:
            raise Error(
                "Bad infectivity",
                "Infectivity function must have a return type `float'",
                f.pos,
            )
    return f


def make_transmissibility(node: PTNode, parent_scope: Scope) -> Function:
    f = Function.make(node, parent_scope)
    match f.params:
        case [Param(_, TypeRef("edge"))]:
            pass
        case _:
            raise Error(
                "Bad transmissibility",
                "Transmissibility function must have one parameter of type `edge'",
                f.pos,
            )
    match f.return_:
        case TypeRef("float"):
            pass
        case _:
            raise Error(
                "Bad transmissibility",
                "Transmissibility function must have a return type `float'",
                f.pos,
            )
    return f


def make_enabled(node: PTNode, parent_scope: Scope) -> Function:
    f = Function.make(node, parent_scope)
    match f.params:
        case [Param(_, TypeRef("edge"))]:
            pass
        case _:
            raise Error(
                "Bad enabled (edge)",
                "Enabled (edge) function must have one parameter of type `edge'",
                f.pos,
            )
    match f.return_:
        case TypeRef("bool"):
            pass
        case _:
            raise Error(
                "Bad enabled (edge)",
                "Enabled (edge) function must have a return type `bool'",
                f.pos,
            )
    return f


def add_builtin_type(type: str, scope: Scope, pos: SourcePosition):
    scope.define(type, BuiltinType(type), pos)


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
    pos: SourcePosition | None = attrs.field(default=None, repr=False, eq=False)

    @classmethod
    def make(cls, module: str, node: PTNode) -> Source:
        scope = Scope(name="source")
        add_builtins(scope, node.pos)

        configs: list[Config] = []
        variables: list[Variable] = []
        enums: list[EnumType] = []
        node_table: UniqueObject[NodeTable] = UniqueObject()
        edge_table: UniqueObject[EdgeTable] = UniqueObject()
        distributions: list[Distribution] = []
        contagions: list[Contagion] = []
        functions: list[Function] = []
        main_function: UniqueObject[Function] = UniqueObject()

        for child in node.named_children:
            match child.type:
                case "config":
                    configs.append(Config.make(child, scope))
                case "variable":
                    variables.append(Variable.make(child, scope))
                case "enum":
                    enums.append(EnumType.make(child, scope))
                case "node":
                    node_table.dup_error(
                        "Multiple node tables",
                        "Node table is multiply defined",
                        child.pos,
                    )
                    node_table.set(NodeTable.make(child, scope))
                case "edge":
                    edge_table.dup_error(
                        "Multiple edge tables",
                        "Edge table is multiply defined",
                        child.pos,
                    )

                    edge_table.set(EdgeTable.make(child, scope))
                case "distributions":
                    for d in child.named_children:
                        distributions.append(parse_distribution(d, scope))
                case "contagion":
                    contagions.append(Contagion.make(child, scope))
                case "function":
                    if child.field("name").text == "main":
                        main_function.dup_error(
                            "Multiple mains",
                            "Main function is multiply defined",
                            child.pos,
                        )
                        main_function.set(make_main(child, scope))
                    else:
                        functions.append(Function.make(child, scope))

        node_table.missing_error(
            "Node undefined", "Node table was not defined", node.pos
        )
        edge_table.missing_error(
            "Edge undefined", "Edge table was not defined", node.pos
        )
        main_function.missing_error(
            "Main undefined", "No main function was defined", node.pos
        )

        node_table.get().link_contagions(contagions)
        edge_table.get().link_contagions(contagions)
        edge_table.get().link_node_table(node_table.get())
        
        for func in functions:
            func.link_tables(node_table.get(), edge_table.get())
        for contagion in contagions:
            contagion.susceptibility.link_tables(node_table.get(), edge_table.get())
            contagion.infectivity.link_tables(node_table.get(), edge_table.get())
            contagion.transmissibility.link_tables(node_table.get(), edge_table.get())
            contagion.enabled.link_tables(node_table.get(), edge_table.get())
        main_function.get().link_tables(node_table.get(), edge_table.get())

        return cls(
            module=module,
            configs=configs,
            variables=variables,
            enums=enums,
            node_table=node_table.get(),
            edge_table=edge_table.get(),
            distributions=distributions,
            contagions=contagions,
            functions=functions,
            main_function=main_function.get(),
            scope=scope,
            pos=node.pos,
        )


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
        case "integer":
            return int(node.text)
        case "float":
            return float(node.text)
        case "boolean":
            return node.text == True
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


Statement = (
    PassStatement
    | ReturnStatement
    | IfStatement
    | WhileLoop
    | CallStatement
    | UpdateStatement
    | PrintStatement
)


StatementParts = Statement | ElseSection | ElifSection | Variable


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


Referable = (
    EnumConstant
    | Config
    | Param
    | Variable
    | BuiltinFunction
    | Function
    | tuple[Contagion, BuiltinFunction]
    | tuple[Contagion, Function]
    | tuple[Param | Variable, NodeField]
    | tuple[Param | Variable, EdgeField]
    | tuple[Param | Variable, Contagion, NodeField]
    | tuple[Param | Variable, SourceNodeAccessor, NodeField]
    | tuple[Param | Variable, SourceNodeAccessor, Contagion, NodeField]
    | tuple[Param | Variable, TargetNodeAccessor, NodeField]
    | tuple[Param | Variable, TargetNodeAccessor, Contagion, NodeField]
)


def resolve_ref(r: Reference) -> Referable:
    assert r.scope is not None
    r_str = ".".join(r.names)

    head, tail = r.names[0], r.names[1:]
    refs = []
    try:
        v = r.scope.resolve(head, r.pos)
        refs.append(v)
        for part in tail:
            v = v.scope.resolve(part, r.pos)
            refs.append(v)
    except (ASTConstructionError, AttributeError) as e:
        raise Error("Undefined reference", f"Failed to resolve '{r_str}'; ({e})", r.pos)

    if len(refs) == 1:
        ret = refs[0]
    else:
        ret = tuple(refs)

    try:
        ret = check_type(ret, Referable)
    except Exception as e:
        raise Error("Invalid reference", f"'{r_str}' is not a valid reference; ({e})", r.pos)

    return ret


def mk_ast1(filename: Path, node: PTNode) -> Source:
    """Make ast1 from parse tree."""
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
