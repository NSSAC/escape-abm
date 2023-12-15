"""Abstract syntax tree v1."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, TypeVar, Generic, Literal, cast, assert_never

import attrs
import click
import rich
import rich.markup
from typeguard import TypeCheckError, check_type

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

        raise Error("Undefined reference", f"{k} is not defined.", pos)

    def define(self, k: str, v: Any, pos: SourcePosition | None):
        if k in self.names:
            raise Error(
                "Already defined",
                f"{k} has been already defined.",
                pos,
            )

        self.names[k] = v


def make_literal(n: PTNode, types: type) -> int | float | bool:
    match n.type:
        case "integer":
            v = int(n.text)
        case "float":
            v = float(n.text)
        case "boolean":
            v = n.text == True
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
            "Expected a built in type.",
            self.pos,
        )

@attrs.define
class BuiltinParam:
    name: str
    type: str

@attrs.define
class BuiltinFunction:
    name: str
    params: list[BuiltinParam]
    return_: str | None
    is_device: bool


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
            "Expected an enumeration constant.",
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
            "Expected an enumerated type.",
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
            "Expected a built in or enumerated type.",
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
class BuiltinConfig:
    name: str
    type: str


@attrs.define
class Global:
    name: str
    type: BuiltinTypeRef
    default: int | float | bool
    pos: SourcePosition | None = attrs.field(default=None, repr=False, eq=False)

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> Global:
        name = node.field("name").text
        type = BuiltinTypeRef.make(node.field("type"), scope)
        default = make_literal(node.field("default"), int | float | bool)
        obj = cls(name, type, default, node.pos)
        scope.define(name, obj, node.pos)
        return obj


@attrs.define
class BuiltinGlobal:
    name: str
    type: str


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
                "Key error",
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
                "Key error",
                "One and only one target node key field must be specified.",
                node.pos,
            )
        target_key = target_key[0]

        source_key = [f for f in fields if f.is_source_node_key]
        if len(source_key) != 1:
            raise Error(
                "Key error",
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
            "Expected a reference to a distribution.",
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
class StateAccessor:
    scope: Scope | None = attrs.field(default=None, repr=False, eq=False)


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
    state: StateAccessor
    step: BuiltinFunction
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
                        "Contagion state type is multiply defined.",
                        child.pos,
                    )
                    state_type.set(EnumTypeRef.make(child.field("type"), scope))
                case "transitions":
                    for grand_child in child.fields("body"):
                        transitions.append(Transition.make(grand_child, scope))
                case "transmissions":
                    for grand_child in child.fields("body"):
                        transmissions.append(Transmission.make(grand_child, scope))
                case "function":
                    match child.field("name").text:
                        case "susceptibility":
                            susceptibility.dup_error(
                                "Multiple susceptibility",
                                "Susceptibility function is multiply defined.",
                                child.pos,
                            )
                            susceptibility.set(make_susceptibility(child, scope))
                        case "infectivity":
                            infectivity.dup_error(
                                "Multiple infectivity",
                                "Infectivity function is multiply defined.",
                                child.pos,
                            )
                            infectivity.set(make_infectivity(child, scope))
                        case "transmissibility":
                            transmissibility.dup_error(
                                "Multiple transmissibility",
                                "Transmissibility function is multiply defined.",
                                child.pos,
                            )
                            transmissibility.set(make_transmissibility(child, scope))
                        case "enabled":
                            enabled.dup_error(
                                "Multiple enabled",
                                "Enabled (edge) function is multiply defined.",
                                child.pos,
                            )
                            enabled.set(make_enabled(child, scope))
                        case _:
                            raise Error(
                                "Invalid function",
                                f"{child.field('name').text} is not a valid contagion function.",
                                child.field("type").pos,
                            )
                case _ as unexpected:
                    raise UnexpectedValue(unexpected, child.pos)

        state_type.missing_error(
            "State type undefined", "State type has not been defined.", node.pos
        )
        susceptibility.missing_error(
            "Susceptibility undefined",
            "Susceptibility function has not been defined.",
            node.pos,
        )
        infectivity.missing_error(
            "Infectivity undefined",
            "Infectivity function has not been defined.",
            node.pos,
        )
        transmissibility.missing_error(
            "Transmissibility undefined",
            "Transmissibility function has not been defined.",
            node.pos,
        )
        enabled.missing_error(
            "Edge enabled undefined",
            "Enabled (edge) function has not been defined.",
            node.pos,
        )

        state = StateAccessor(scope)
        scope.define("state", state, None)

        step = BuiltinFunction(
            "step",
            [
                BuiltinParam("elapsed", "float"),
            ],
            None,
            True,
        )
        scope.define("step", step, None)

        obj = cls(
            name=name,
            state_type=state_type.get(),
            transitions=transitions,
            transmissions=transmissions,
            susceptibility=susceptibility.get(),
            infectivity=infectivity.get(),
            transmissibility=transmissibility.get(),
            enabled=enabled.get(),
            state=state,
            step=step,
            scope=scope,
            pos=node.pos,
        )
        parent_scope.define(name, obj, node.pos)
        return obj


@attrs.define
class NodeSet:
    name: str
    pos: SourcePosition | None = attrs.field(default=None, repr=False, eq=False)

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> NodeSet:
        name = node.text
        obj = cls(name, node.pos)
        scope.define(name, obj, node.pos)
        return obj


@attrs.define
class EdgeSet:
    name: str
    pos: SourcePosition | None = attrs.field(default=None, repr=False, eq=False)

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> EdgeSet:
        name = node.text
        obj = cls(name, node.pos)
        scope.define(name, obj, node.pos)
        return obj


@attrs.define
class SetRef:
    name: str
    scope: Scope | None = attrs.field(default=None, repr=False, eq=False)
    pos: SourcePosition | None = attrs.field(default=None, repr=False, eq=False)

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> SetRef:
        return cls(node.text, scope, node.pos)

    def resolve(self) -> NodeSet | EdgeSet:
        assert self.scope is not None

        v = self.scope.resolve(self.name, self.pos)
        if isinstance(v, NodeSet | EdgeSet):
            return v

        raise Error(
            "Invalid value",
            "Expected a reference to a nodeset or edgeset.",
            self.pos,
        )


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

    @property
    def names_str(self) -> str:
        return ".".join(self.names)

    def do_resolve(self) -> Any:
        assert self.scope is not None

        head, tail = self.names[0], self.names[1:]
        refs = []
        try:
            v = self.scope.resolve(head, self.pos)
            refs.append(v)
            for part in tail:
                v = v.scope.resolve(part, self.pos)
                refs.append(v)
        except (ASTConstructionError, AttributeError) as e:
            raise Error(
                "Undefined reference",
                f"Failed to resolve '{self.names_str}'; ({e})",
                self.pos,
            )

        if len(refs) == 1:
            ret = refs[0]
        else:
            ret = tuple(refs)

        return ret

    def resolve_referable(self) -> Referable:
        ret = self.do_resolve()
        try:
            ret = check_type(ret, Referable)
        except TypeCheckError:
            raise Error(
                "Invalid reference",
                f"'{self.names_str}' is not a valid referable",
                self.pos,
            )
        return ret

    def resolve_callable(self) -> Callable:
        ret = self.do_resolve()
        try:
            ret = check_type(ret, Callable)
        except TypeCheckError:
            raise Error(
                "Invalid reference",
                f"'{self.names_str}' is not a valid callable",
                self.pos,
            )
        return ret

    def resolve_updateable(self) -> Updateable:
        ret = self.do_resolve()
        try:
            ret = check_type(ret, Updateable)
        except TypeCheckError:
            raise Error(
                "Invalid reference",
                f"'{self.names_str}' can't be updated",
                self.pos,
            )
        return ret


@attrs.define
class FunctionCall:
    function: Reference
    args: list[Expression]
    pos: SourcePosition | None = attrs.field(default=None, repr=False, eq=False)

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> FunctionCall:
        function = Reference.make(node.field("function"), scope)
        args = [parse_expression(c, scope) for c in node.fields("arg")]

        r_str = ".".join(function.names)
        call_name = f"_function_call_{r_str}_{node.pos.line}_{node.pos.col}"
        obj = cls(function, args, node.pos)
        scope.define(call_name, obj, node.pos)
        return obj

    def type_check(self):
        function = self.function.resolve_callable()

        match function:
            case BuiltinFunction() | Function():
                n_params = len(function.params)
                n_args = len(self.args)
                if n_params != n_args:
                    raise Error(
                        "Invalid args",
                        f"Expected {n_params} args; got {n_args}.",
                        self.pos,
                    )
            case [_, BuiltinFunction() | Function() as f]:
                n_params = len(f.params)
                n_args = len(self.args)
                if n_params != n_args:
                    raise Error(
                        "Invalid args",
                        f"Expected {n_params} args; got {n_args}.",
                        self.pos,
                    )
            case _ as unexpected:
                assert_never(unexpected)


@attrs.define
class Variable:
    name: str
    type: TypeRef
    init: Expression
    scope: Scope | None = attrs.field(default=None, repr=False, eq=False)
    pos: SourcePosition | None = attrs.field(default=None, repr=False, eq=False)

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> Variable:
        var = node.field("name").text
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
class SelectUsing:
    name: str
    set: SetRef
    function: FunctionRef
    pos: SourcePosition | None = attrs.field(default=None, repr=False, eq=False)

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> SelectUsing:
        set = SetRef.make(node.field("set"), scope)
        function = FunctionRef.make(node.field("function"), scope)
        name = f"select_{set.name}_using__{function.name}_{node.pos.line}"
        obj = cls(name, set, function, node.pos)
        scope.define(name, obj, node.pos)
        return obj

    def type_check(self):
        func = self.function.resolve()
        match self.set.resolve():
            case NodeSet():
                if not is_node_func(func):
                    raise Error(
                        "Invalid function type",
                        "Nodeset update functions must have a single parameter of type node.",
                        self.pos,
                    )
            case EdgeSet():
                if not is_edge_func(func):
                    raise Error(
                        "Invalid function type",
                        "Edgeset update functions must have a single parameter of type edge.",
                        self.pos,
                    )
        if not is_bool_func(func):
            raise Error(
                "Invalid function type",
                "Sets can can only be updated using functions that return a single bool value.",
                self.pos,
            )


@attrs.define
class SelectApprox:
    name: str
    set: SetRef
    amount: int | float
    parent: SetRef
    pos: SourcePosition | None = attrs.field(default=None, repr=False, eq=False)

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> SelectApprox:
        set = SetRef.make(node.field("set"), scope)
        amount = make_literal(node.field("amount"), int | float)
        parent = SetRef.make(node.field("parent"), scope)
        name = f"select_{set.name}_from_{parent.name}_approx_{node.pos.line}"
        obj = cls(name, set, amount, parent, node.pos)
        scope.define(name, obj, node.pos)
        return obj

    def type_check(self):
        set_type = type(self.set.resolve())
        parent_type = type(self.parent.resolve())
        if set_type == parent_type:
            return

        if set_type == NodeSet:
            raise Error(
                "Invalid source set type",
                "Nodesets can only be sampled from other nodesets.",
                self.pos,
            )
        elif set_type == EdgeSet:
            raise Error(
                "Invalid source set type",
                "Edgesets can only be sampled from other edgesets.",
                self.pos,
            )
        else:
            raise TypeError("Unexpected type %r" % set_type)


@attrs.define
class SelectRelative:
    name: str
    set: SetRef
    amount: int | float
    parent: SetRef
    pos: SourcePosition | None = attrs.field(default=None, repr=False, eq=False)

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> SelectRelative:
        set = SetRef.make(node.field("set"), scope)
        amount = make_literal(node.field("amount"), int | float)
        parent = SetRef.make(node.field("parent"), scope)
        name = f"select_{set.name}_from_{parent.name}_relative_{node.pos.line}"
        obj = cls(name, set, amount, parent, node.pos)
        scope.define(name, obj, node.pos)
        return obj

    def type_check(self):
        set_type = type(self.set.resolve())
        parent_type = type(self.parent.resolve())
        if set_type == parent_type:
            return

        if set_type == NodeSet:
            raise Error(
                "Invalid source set type",
                "Nodesets can only be sampled from other nodesets.",
                self.pos,
            )
        elif set_type == EdgeSet:
            raise Error(
                "Invalid source set type",
                "Edgesets can only be sampled from other edgesets.",
                self.pos,
            )
        else:
            raise TypeError("Unexpected type %r" % set_type)


@attrs.define
class ForeachStatement:
    name: str
    type: Literal["node", "edge"]
    set: SetRef
    function: FunctionRef
    pos: SourcePosition | None = attrs.field(default=None, repr=False, eq=False)

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> ForeachStatement:
        type = node.field("type").text
        type = cast(Literal["node", "edge"], type)
        set = SetRef.make(node.field("set"), scope)
        function = FunctionRef.make(node.field("function"), scope)
        name = f"foreach_{type}_in_{set.name}_run_{function.name}_{node.pos.line}"
        obj = cls(name, type, set, function, node.pos)
        scope.define(name, obj, node.pos)
        return obj

    def type_check(self):
        set = self.set.resolve()
        function = self.function.resolve()

        match self.type:
            case "node":
                if not isinstance(set, NodeSet):
                    raise Error(
                        "Invalid set type",
                        "Foreach node statement must be run on nodesets.",
                        self.pos,
                    )
                if not is_node_func(function):
                    raise Error(
                        "Invalid function type",
                        "Foreach node statements must run functions with a single parameter of type node.",
                        self.pos,
                    )
            case "edge":
                if not isinstance(set, EdgeSet):
                    raise Error(
                        "Invalid set type",
                        "Foreach edge statement must be run on edgesets.",
                        self.pos,
                    )
                if not is_node_func(function):
                    raise Error(
                        "Invalid function type",
                        "Foreach edge statements must run functions with a single parameter of type edge.",
                        self.pos,
                    )
            case _ as unexpected:
                assert_never(unexpected)

        if not is_void_func(function):
            raise Error(
                "Invalid function type",
                "Foreach statements must run functions that do not return values.",
                self.pos,
            )


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
    return_: TypeRef | None
    body: list[Statement]
    scope: Scope | None = attrs.field(default=None, repr=False, eq=False)
    pos: SourcePosition | None = attrs.field(default=None, repr=False, eq=False)

    @classmethod
    def make(cls, node: PTNode, parent_scope: Scope) -> Function:
        name = node.field("name").text
        scope = Scope(name=name, parent=parent_scope)

        params: list[Param] = []
        for child in node.field("params").named_children:
            params.append(Param.make(child, scope))

        return_node = node.maybe_field("return")
        if return_node is None:
            return_ = None
        else:
            return_ = TypeRef.make(return_node, scope)

        body: list[Statement] = []
        for child in node.field("body").named_children:
            stmt = parse_statement(child, scope)
            body.append(stmt)

        obj = cls(name, params, return_, body, scope, node.pos)
        parent_scope.define(name, obj, node.pos)
        return obj

    def variables(self) -> list[Variable]:
        assert self.scope is not None
        return [s for s in self.scope.names.values() if isinstance(s, Variable)]

    def device_statements(self) -> list[DeviceStatement]:
        assert self.scope is not None
        return [s for s in self.scope.names.values() if isinstance(s, DeviceStatement)]

    def function_calls(self) -> list[FunctionCall]:
        assert self.scope is not None
        return [s for s in self.scope.names.values() if isinstance(s, FunctionCall)]

    def link_tables(self, node_table: NodeTable, edge_table: EdgeTable):
        for param in self.params:
            param.link_tables(node_table, edge_table)
        for var in self.variables():
            var.link_tables(node_table, edge_table)

    def type_check(self):
        for s in self.device_statements():
            s.type_check()
        for c in self.function_calls():
            c.type_check()


@attrs.define
class FunctionRef:
    name: str
    scope: Scope | None = attrs.field(default=None, repr=False, eq=False)
    pos: SourcePosition | None = attrs.field(default=None, repr=False, eq=False)

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> FunctionRef:
        return cls(node.text, scope, node.pos)

    def resolve(self) -> Function:
        assert self.scope is not None

        v = self.scope.resolve(self.name, self.pos)
        if isinstance(v, Function):
            return v

        raise Error(
            "Invalid value",
            "Expected a reference to a function",
            self.pos,
        )


def is_node_func(f: Function) -> bool:
    match f.params:
        case [Param(_, TypeRef("node"))]:
            return True
        case _:
            return False


def is_edge_func(f: Function) -> bool:
    match f.params:
        case [Param(_, TypeRef("edge"))]:
            return True
        case _:
            return False


def is_int_func(f: Function) -> bool:
    match f.return_:
        case TypeRef("int"):
            return True
        case _:
            return False


def is_float_func(f: Function) -> bool:
    match f.return_:
        case TypeRef("float"):
            return True
        case _:
            return False


def is_bool_func(f: Function) -> bool:
    match f.return_:
        case TypeRef("bool"):
            return True
        case _:
            return False


def is_void_func(f: Function) -> bool:
    return f.return_ is None


def is_kernel_func(f: Function) -> bool:
    if f.device_statements():
        return False
    else:
        return True


def make_initialize(node: PTNode, parent_scope: Scope) -> Function:
    f = Function.make(node, parent_scope)
    if f.params or f.return_ is not None:
        raise Error(
            "Bad initialize", "Initialize function must not have any parameters.", f.pos
        )
    if f.return_ is not None:
        raise Error(
            "Bad initialize", "Initialize function must not have a return type.", f.pos
        )
    return f


def make_intervene(node: PTNode, parent_scope: Scope) -> Function:
    f = Function.make(node, parent_scope)
    if f.params or f.return_ is not None:
        raise Error(
            "Bad intervene", "Intervene function must not have any parameters.", f.pos
        )
    if f.return_ is not None:
        raise Error(
            "Bad intervene", "Intervene function must not have a return type.", f.pos
        )
    return f


def make_susceptibility(node: PTNode, parent_scope: Scope) -> Function:
    f = Function.make(node, parent_scope)
    if not is_node_func(f):
        raise Error(
            "Bad susceptibility",
            "Susceptibility function must have one parameter of type `node'",
            f.pos,
        )
    if not is_float_func(f):
        raise Error(
            "Bad susceptibility",
            "Susceptibility function must have a return type `float'.",
            f.pos,
        )
    if not is_kernel_func(f):
        raise Error(
            "Bad susceptibility",
            "Susceptibility function must not have parallel statements.",
            f.pos,
        )
    return f


def make_infectivity(node: PTNode, parent_scope: Scope) -> Function:
    f = Function.make(node, parent_scope)
    if not is_node_func(f):
        raise Error(
            "Bad infectivity",
            "Infectivity function must have one parameter of type `node'",
            f.pos,
        )
    if not is_float_func(f):
        raise Error(
            "Bad infectivity",
            "Infectivity function must have a return type `float'",
            f.pos,
        )
    if not is_kernel_func(f):
        raise Error(
            "Bad infectivity",
            "Infectivity function must not have parallel statements.",
            f.pos,
        )
    return f


def make_transmissibility(node: PTNode, parent_scope: Scope) -> Function:
    f = Function.make(node, parent_scope)
    if not is_edge_func(f):
        raise Error(
            "Bad transmissibility",
            "Transmissibility function must have one parameter of type `edge'",
            f.pos,
        )
    if not is_float_func(f):
        raise Error(
            "Bad transmissibility",
            "Transmissibility function must have a return type `float'",
            f.pos,
        )
    if not is_kernel_func(f):
        raise Error(
            "Bad transmissibility",
            "Transmissibility function must not have parallel statements.",
            f.pos,
        )
    return f


def make_enabled(node: PTNode, parent_scope: Scope) -> Function:
    f = Function.make(node, parent_scope)
    if not is_edge_func(f):
        raise Error(
            "Bad enabled (edge)",
            "Enabled (edge) function must have one parameter of type `edge'",
            f.pos,
        )
    if not is_bool_func(f):
        raise Error(
            "Bad enabled (edge)",
            "Enabled (edge) function must have a return type `bool'",
            f.pos,
        )
    if not is_kernel_func(f):
        raise Error(
            "Bad enabled (edge)",
            "Enabled (edge) function must not have parallel statements.",
            f.pos,
        )
    return f


def add_builtins(scope: Scope, pos: SourcePosition):
    # fmt: off
    builtin_types = [
        "int", "uint", "float", "bool", "size",
        "node", "edge",
        "nodeset", "edgeset",
        "u8", "u16", "u32", "u64",
        "i8", "i16", "i32", "i64",
        "f32", "f64",
    ]
    # fmt: on

    for type in builtin_types:
        scope.define(type, BuiltinType(type), pos)

    scope.define("num_ticks", BuiltinConfig("num_ticks", "uint"), pos)

    scope.define("cur_tick", BuiltinGlobal("cur_tick", "uint"), pos)
    scope.define("num_nodes", BuiltinGlobal("num_nodes", "size"), pos)
    scope.define("num_edges", BuiltinGlobal("num_edges", "size"), pos)


@attrs.define
class Source:
    module: str
    enums: list[EnumType]
    configs: list[Config]
    globals: list[Global]
    node_table: NodeTable
    edge_table: EdgeTable
    distributions: list[Distribution]
    contagions: list[Contagion]
    nodesets: list[NodeSet]
    edgesets: list[EdgeSet]
    functions: list[Function]
    initialize: Function
    intervene: Function
    scope: Scope | None = attrs.field(default=None, repr=False, eq=False)
    pos: SourcePosition | None = attrs.field(default=None, repr=False, eq=False)

    @classmethod
    def make(cls, module: str, node: PTNode) -> Source:
        scope = Scope(name="source")
        add_builtins(scope, node.pos)

        enums: list[EnumType] = []
        configs: list[Config] = []
        globals: list[Global] = []
        node_table: UniqueObject[NodeTable] = UniqueObject()
        edge_table: UniqueObject[EdgeTable] = UniqueObject()
        distributions: list[Distribution] = []
        contagions: list[Contagion] = []
        nodesets: list[NodeSet] = []
        edgesets: list[EdgeSet] = []
        functions: list[Function] = []
        initialize: UniqueObject[Function] = UniqueObject()
        intervene: UniqueObject[Function] = UniqueObject()

        for child in node.named_children:
            match child.type:
                case "enum":
                    enums.append(EnumType.make(child, scope))
                case "config":
                    configs.append(Config.make(child, scope))
                case "global":
                    globals.append(Global.make(child, scope))
                case "node":
                    node_table.dup_error(
                        "Multiple node tables",
                        "Node table is multiply defined.",
                        child.pos,
                    )
                    node_table.set(NodeTable.make(child, scope))
                case "edge":
                    edge_table.dup_error(
                        "Multiple edge tables",
                        "Edge table is multiply defined.",
                        child.pos,
                    )
                    edge_table.set(EdgeTable.make(child, scope))
                case "distributions":
                    for grand_child in child.named_children:
                        distributions.append(parse_distribution(grand_child, scope))
                case "contagion":
                    contagions.append(Contagion.make(child, scope))
                case "nodeset":
                    for grand_child in child.fields("name"):
                        nodesets.append(NodeSet.make(grand_child, scope))
                case "edgeset":
                    for grand_child in child.fields("name"):
                        edgesets.append(EdgeSet.make(grand_child, scope))
                case "function":
                    if child.field("name").text == "initialize":
                        initialize.dup_error(
                            "Multiple initialize",
                            "Initialize function is multiply defined.",
                            child.pos,
                        )
                        initialize.set(make_initialize(child, scope))
                    elif child.field("name").text == "intervene":
                        intervene.dup_error(
                            "Multiple intervene",
                            "Intervene function is multiply defined.",
                            child.pos,
                        )
                        intervene.set(make_intervene(child, scope))
                    else:
                        functions.append(Function.make(child, scope))

        node_table.missing_error(
            "Node undefined", "Node table was not defined.", node.pos
        )
        edge_table.missing_error(
            "Edge undefined", "Edge table was not defined.", node.pos
        )
        initialize.missing_error(
            "Initialize undefined", "No initialize function was defined.", node.pos
        )
        intervene.missing_error(
            "Intervene undefined", "No intervene function was defined.", node.pos
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
        initialize.get().link_tables(node_table.get(), edge_table.get())
        intervene.get().link_tables(node_table.get(), edge_table.get())

        for contagion in contagions:
            contagion.susceptibility.type_check()
            contagion.infectivity.type_check()
            contagion.transmissibility.type_check()
            contagion.enabled.type_check()
        for func in functions:
            func.type_check()
        initialize.get().type_check()
        intervene.get().type_check()

        return cls(
            module=module,
            enums=enums,
            configs=configs,
            globals=globals,
            node_table=node_table.get(),
            edge_table=edge_table.get(),
            distributions=distributions,
            contagions=contagions,
            nodesets=nodesets,
            edgesets=edgesets,
            functions=functions,
            initialize=initialize.get(),
            intervene=intervene.get(),
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


HostStatement = (
    PassStatement
    | ReturnStatement
    | IfStatement
    | WhileLoop
    | CallStatement
    | UpdateStatement
    | PrintStatement
    | Variable
)

DeviceStatement = SelectUsing | SelectApprox | SelectRelative | ForeachStatement

Statement = HostStatement | DeviceStatement

StatementParts = Statement | ElseSection | ElifSection


def parse_statement(node: PTNode, scope: Scope) -> Statement:
    match node.type:
        case "variable":
            return Variable.make(node, scope)
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
        case "select_using":
            return SelectUsing.make(node, scope)
        case "select_relative":
            return SelectRelative.make(node, scope)
        case "select_approx":
            return SelectApprox.make(node, scope)
        case "foreach_statement":
            return ForeachStatement.make(node, scope)
        case unexpected:
            raise UnexpectedValue(unexpected, node.pos)


Referable = (
    EnumConstant
    | Config
    | BuiltinConfig
    | Global
    | BuiltinGlobal
    | Param
    | Variable
    | NodeSet
    | EdgeSet
    | tuple[Param | Variable, NodeField]
    | tuple[Param | Variable, EdgeField]
    | tuple[Param | Variable, Contagion, StateAccessor]
    | tuple[Param | Variable, SourceNodeAccessor]
    | tuple[Param | Variable, TargetNodeAccessor]
    | tuple[Param | Variable, SourceNodeAccessor, NodeField]
    | tuple[Param | Variable, TargetNodeAccessor, NodeField]
)

Callable = (
    BuiltinFunction
    | Function
    | tuple[Contagion, BuiltinFunction]
    | tuple[Contagion, Function]
)

Updateable = (
    Global
    | Param
    | Variable
    | tuple[Param | Variable, NodeField]
    | tuple[Param | Variable, EdgeField]
    | tuple[Param | Variable, Contagion, StateAccessor]
    | tuple[Param | Variable, SourceNodeAccessor, NodeField]
    | tuple[Param | Variable, TargetNodeAccessor, NodeField]
)


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
        sys.exit(1)
