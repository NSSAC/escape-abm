"""Abstract syntax tree."""

from __future__ import annotations

from pathlib import Path
from collections import defaultdict
from contextlib import contextmanager
from typing import (
    Any,
    Callable,
    Generator,
    Literal,
    TypeVar,
    cast,
    overload,
)

import rich
import rich.markup
import click
from rich.tree import Tree
from rich.pretty import Pretty
from pydantic import BaseModel, Field

from .misc import SourcePosition, EslError, RichException
from .parse_tree import mk_pt, PTNode
from .click_helpers import simulation_file_option

Type = TypeVar("Type")


@contextmanager
def err_desc(description: str, pos: SourcePosition | None):
    try:
        yield
    except AssertionError:
        raise EslError("Semantic error", description, pos)


class Scope(BaseModel):
    name: str
    names: dict[str, Any] = Field(default_factory=dict)
    parent: Scope | None = Field(default=None)
    children: list[Scope] = Field(default_factory=list)

    def model_post_init(self, _: Any) -> None:
        if self.parent is not None:
            self.parent.children.append(self)

    def __rich_repr__(self):
        yield "name", self.name
        yield "names", {k: type(v).__name__ for k, v in self.names.items()}
        if self.parent is not None:
            yield "parent", self.parent.name

    def resolve(self, k: str) -> Any:
        if k in self.names:
            return self.names[k]
        elif self.parent is not None:
            return self.parent.resolve(k)

        raise KeyError(f"{k} is not defined.")

    def define(self, k: str, v: Any):
        if k in self.names:
            raise ValueError(f"{k} has been already defined.")
        self.names[k] = v

    def root(self) -> Scope:
        ret = self
        while ret.parent is not None:
            ret = ret.parent
        return ret

    def visit(self) -> Generator[tuple[str, Any], None, None]:
        yield from self.names.items()
        for child in self.children:
            yield from child.visit()

    def rich_tree(self, tree: Tree | None = None) -> Tree:
        if tree is None:
            tree = Tree(Pretty(self))
        else:
            tree = tree.add(Pretty(self))

        for child in self.children:
            child.rich_tree(tree)

        return tree


NodeParser = Callable[[PTNode, Scope], Any]
_NODE_PARSER: dict[str, NodeParser] = {}


def parse(node: PTNode, scope: Scope) -> Any:
    try:
        node_parser = _NODE_PARSER[node.type]
    except KeyError:
        raise EslError(
            "AST Error",
            f"Parser for node type '{node.type}' is not defined",
            node.pos,
        )

    try:
        return node_parser(node, scope)
    except EslError:
        raise
    except Exception as e:
        estr = f"{e.__class__.__name__}: {e!s}"
        raise EslError(
            "AST Error", f"Failed to parse '{node.type}': ({estr})", node.pos
        )


def register_parser(type: str, parser: NodeParser):
    if type in _NODE_PARSER:
        raise ValueError(f"Parser for '{type}' is already defined")
    _NODE_PARSER[type] = parser


register_parser("integer", lambda node, _: int(node.text))
register_parser("float", lambda node, _: float(node.text))
register_parser("boolean", lambda node, _: node.text == "True")
register_parser("string", lambda node, _: node.text)


_INSTANCE_REGISTRY: dict[type, list[Any]] = defaultdict(list)


def register_instance(o: Any) -> None:
    k = type(o)
    _INSTANCE_REGISTRY[k].append(o)


def get_instances(k: type) -> list[Any]:
    return _INSTANCE_REGISTRY[k]


def clear_instances() -> None:
    for v in _INSTANCE_REGISTRY.values():
        v.clear()


class Reference(BaseModel):
    names: list[str]
    scope: Scope = Field(repr=False)
    pos: SourcePosition | None = Field(default=None, repr=False)
    value_: Any | None = Field(default=None, repr=False)

    def model_post_init(self, _: Any) -> None:
        register_instance(self)

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> Reference:
        names = [s.text for s in node.named_children]
        obj = cls(names=names, scope=scope, pos=node.pos)
        return obj

    @property
    def name(self) -> str:
        return ".".join(self.names)

    @property
    def value(self) -> Any:
        if self.value_ is not None:
            return self.value_

        head, tail = self.names[0], self.names[1:]
        refs = []
        try:
            v = self.scope.resolve(head)
            refs.append(v)
            for part in tail:
                v = v.scope.resolve(part)
                refs.append(v)
        except Exception as e:
            raise EslError(
                "Reference error", f"Failed to resolve '{self.name}'; ({e!s})", self.pos
            )

        if len(refs) == 1:
            value = refs[0]
        else:
            value = tuple(refs)

        self.value_ = value
        return self.value_


register_parser("reference", Reference.make)


class TemplateVariable(BaseModel):
    pos: SourcePosition | None = Field(default=None, repr=False)

    def model_post_init(self, _: Any) -> None:
        register_instance(self)

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> TemplateVariable:
        scope = scope
        return cls(pos=node.pos)


register_parser("template_variable", TemplateVariable.make)


class BuiltinType(BaseModel):
    name: str


class BuiltinParam(BaseModel):
    name: str
    type: str


class BuiltinFunction(BaseModel):
    name: str
    params: list[BuiltinParam]
    rtype: str | None


class BuiltinGlobal(BaseModel):
    name: str
    type: str
    is_config: bool


class BuiltinNodeset(BaseModel):
    name: str


class BuiltinEdgeset(BaseModel):
    name: str


class EnumConstant(BaseModel):
    name: str
    type: EnumType


class EnumType(BaseModel):
    name: str
    consts: list[str]
    pos: SourcePosition | None = Field(default=None, repr=False)

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> EnumType:
        name = node.field("name").text
        consts = [child.text for child in node.fields("const")]

        obj = cls(name=name, consts=consts)
        for child in node.fields("const"):
            enum_const = EnumConstant(name=child.text, type=obj)
            scope.define(enum_const.name, enum_const)

        scope.define(name, obj)
        return obj


register_parser("enum", EnumType.make)


class Global(BaseModel):
    name: str
    type: Reference
    is_config: bool
    default: int | float | bool
    pos: SourcePosition | None = Field(default=None, repr=False)

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> Global:
        name = node.field("name").text
        is_config = node.field("category").text == "config"
        type = parse(node.field("type"), scope)
        default = parse(node.field("default"), scope)
        obj = cls(
            name=name, type=type, is_config=is_config, default=default, pos=node.pos
        )
        scope.define(name, obj)
        return obj


register_parser("global", Global.make)


class NodeField(BaseModel):
    name: str
    type: Reference
    is_node_key: bool
    is_static: bool
    save_to_output: bool
    pos: SourcePosition | None = Field(default=None, repr=False)

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> NodeField:
        name = node.field("name").text
        type = parse(node.field("type"), scope)

        annotations = [c.text for c in node.fields("annotation")]
        is_node_key = "node key" in annotations
        is_static = "static" in annotations or is_node_key
        save_to_output = "save" in annotations

        with err_desc("Static field can't be saved to output.", node.pos):
            assert not (save_to_output and is_static)

        obj = cls(
            name=name,
            type=type,
            is_node_key=is_node_key,
            is_static=is_static,
            save_to_output=save_to_output,
            pos=node.pos,
        )
        if not is_node_key:
            scope.define(name, obj)
        return obj

    @property
    def is_used(self):
        return not self.is_node_key


register_parser("node_field", NodeField.make)


class NodeTable(BaseModel):
    fields: list[NodeField]
    key: NodeField
    contagions: list[Contagion] = Field(default=[])
    scope: Scope | None = Field(default=None, repr=False)
    pos: SourcePosition | None = Field(default=None, repr=False)

    @classmethod
    def make(cls, node: PTNode, parent_scope: Scope) -> NodeTable:
        scope = Scope(name="node_table", parent=parent_scope)

        fields = [parse(child, scope) for child in node.named_children]
        key = [f for f in fields if f.is_node_key]
        with err_desc("One and only one node key field must be specified.", node.pos):
            assert len(key) == 1
        key = key[0]

        obj = cls(fields=fields, key=key, contagions=[], scope=scope, pos=node.pos)
        return obj

    def link_contagions(self, contagions: list[Contagion]):
        assert self.scope is not None

        for contagion in contagions:
            self.scope.define(contagion.name, contagion)
            self.contagions.append(contagion)


register_parser("node", NodeTable.make)


class EdgeField(BaseModel):
    name: str
    type: Reference
    is_target_node_key: bool
    is_source_node_key: bool
    is_static: bool
    save_to_output: bool
    pos: SourcePosition | None = Field(default=None, repr=False)

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> EdgeField:
        name = node.field("name").text
        type = parse(node.field("type"), scope)

        annotations = [c.text for c in node.fields("annotation")]
        is_source_node_key = "source node key" in annotations
        is_target_node_key = "target node key" in annotations
        is_static = "static" in annotations or is_source_node_key or is_target_node_key
        save_to_output = "save" in annotations

        with err_desc("Static field can't be saved to output.", node.pos):
            assert not (save_to_output and is_static)

        obj = cls(
            name=name,
            type=type,
            is_target_node_key=is_target_node_key,
            is_source_node_key=is_source_node_key,
            is_static=is_static,
            save_to_output=save_to_output,
            pos=node.pos,
        )
        if not (is_target_node_key or is_source_node_key):
            scope.define(name, obj)
        return obj

    @property
    def is_used(self):
        return not (self.is_target_node_key or self.is_source_node_key)


register_parser("edge_field", EdgeField.make)


class TargetNodeAccessor(BaseModel):
    scope: Scope | None = Field(default=None, repr=False)


class SourceNodeAccessor(BaseModel):
    scope: Scope | None = Field(default=None, repr=False)


class EdgeTable(BaseModel):
    fields: list[EdgeField]
    target_node_key: EdgeField
    source_node_key: EdgeField
    contagions: list[Contagion]
    source_node: SourceNodeAccessor
    target_node: TargetNodeAccessor
    scope: Scope | None = Field(default=None, repr=False)
    pos: SourcePosition | None = Field(default=None, repr=False)

    @classmethod
    def make(cls, node: PTNode, parent_scope: Scope) -> EdgeTable:
        scope = Scope(name="edge_table", parent=parent_scope)

        fields = [parse(child, scope) for child in node.named_children]
        target_key = [f for f in fields if f.is_target_node_key]
        with err_desc(
            "One and only one target node key field must be specified.", node.pos
        ):
            assert len(target_key) == 1
        target_key = target_key[0]

        source_key = [f for f in fields if f.is_source_node_key]
        with err_desc(
            "One and only one source node key field must be specified.", node.pos
        ):
            assert len(source_key) == 1
        source_key = source_key[0]

        source_node = SourceNodeAccessor()
        scope.define("source_node", source_node)

        target_node = TargetNodeAccessor()
        scope.define("target_node", target_node)

        obj = cls(
            fields=fields,
            target_node_key=target_key,
            source_node_key=source_key,
            contagions=[],
            source_node=source_node,
            target_node=target_node,
            scope=scope,
            pos=node.pos,
        )
        return obj

    def link_node_table(self, node_table: NodeTable):
        self.source_node.scope = node_table.scope
        self.target_node.scope = node_table.scope

    def link_contagions(self, contagions: list[Contagion]):
        assert self.scope is not None

        for contagion in contagions:
            self.scope.define(contagion.name, contagion)
            self.contagions.append(contagion)


register_parser("edge", EdgeTable.make)


class DiscreteDist(BaseModel):
    name: str
    ps: list[int | float]
    vs: list[Expression]
    pos: SourcePosition | None = Field(default=None, repr=False)

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> DiscreteDist:
        name = node.field("name").text
        ps = []
        vs = []
        for child in node.fields("pv"):
            ps.append(parse(child.field("p"), scope))
            vs.append(parse(child.field("v"), scope))

        obj = cls(name=name, ps=ps, vs=vs, pos=node.pos)
        scope.define(name, obj)
        return obj


register_parser("discrete_dist", DiscreteDist.make)


class NormalDist(BaseModel):
    name: str
    mean: Expression
    std: Expression
    min: Expression
    max: Expression
    pos: SourcePosition | None = Field(default=None, repr=False)

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> NormalDist:
        name = node.field("name").text
        mean = parse(node.field("mean"), scope)
        std = parse(node.field("std"), scope)

        min = node.maybe_field("min")
        min = -1e300 if min is None else parse(min, scope)

        max = node.maybe_field("max")
        max = 1e300 if max is None else parse(max, scope)

        obj = cls(name=name, mean=mean, std=std, min=min, max=max, pos=node.pos)
        scope.define(name, obj)
        return obj


register_parser("normal_dist", NormalDist.make)


class UniformDist(BaseModel):
    name: str
    low: Expression
    high: Expression
    pos: SourcePosition | None = Field(default=None, repr=False)

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> UniformDist:
        name = node.field("name").text
        low = parse(node.field("low"), scope)
        high = parse(node.field("high"), scope)
        obj = cls(name=name, low=low, high=high, pos=node.pos)
        scope.define(name, obj)
        return obj


register_parser("uniform_dist", UniformDist.make)


Distribution = DiscreteDist | NormalDist | UniformDist


def parse_distributions(node: PTNode, scope: Scope) -> list[Distribution]:
    return [parse(child, scope) for child in node.named_children]


register_parser("distributions", parse_distributions)


@overload
def if_fn_make_ref(x: Function, scope: Scope) -> Reference: ...


@overload
def if_fn_make_ref(x: Type, scope: Scope) -> Type: ...


def if_fn_make_ref(x: Any, scope: Scope) -> Any:
    if isinstance(x, Function):
        return Reference(names=[x.name], scope=scope, pos=x.pos)
    else:
        return x


class Transition(BaseModel):
    entry: Reference
    exit: Reference
    pexpr: Expression
    dwell: Expression
    pos: SourcePosition | None = Field(default=None, repr=False)

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> Transition:
        entry = parse(node.field("entry"), scope)
        exit = parse(node.field("exit"), scope)
        pexpr = node.maybe_field("p")
        if pexpr is not None:
            pexpr = parse(pexpr, scope)
            pexpr = if_fn_make_ref(pexpr, scope)
        else:
            pexpr = 1
        dwell = parse(node.field("dwell"), scope)
        dwell = if_fn_make_ref(dwell, scope)
        return cls(entry=entry, exit=exit, pexpr=pexpr, dwell=dwell, pos=node.pos)


register_parser("transition", Transition.make)


def parse_transitions(node: PTNode, scope: Scope) -> list[Transition]:
    return [parse(child, scope) for child in node.fields("body")]


register_parser("transitions", parse_transitions)


class Transmission(BaseModel):
    contact: Reference
    entry: Reference
    exit: Reference
    pos: SourcePosition | None = Field(default=None, repr=False)

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> Transmission:
        contact = parse(node.field("contact"), scope)
        entry = parse(node.field("entry"), scope)
        exit = parse(node.field("exit"), scope)
        return cls(contact=contact, entry=entry, exit=exit, pos=node.pos)


register_parser("transmission", Transmission.make)


def parse_transmissions(node: PTNode, scope: Scope) -> list[Transmission]:
    return [parse(child, scope) for child in node.fields("body")]


register_parser("transmissions", parse_transmissions)


def parse_contagion_state_type(node: PTNode, scope: Scope) -> Reference:
    return parse(node.field("type"), scope)


register_parser("contagion_state_type", parse_contagion_state_type)


def parse_contagion_function(node: PTNode, scope: Scope) -> tuple[str, Reference]:
    type = node.field("type").text
    function = parse(node.field("function"), scope)
    return type, function


register_parser("contagion_function", parse_contagion_function)


class StateAccessor(BaseModel):
    scope: Scope | None = Field(default=None, repr=False)


class Contagion(BaseModel):
    name: str
    state_type: Reference
    transitions: list[Transition]
    transmissions: list[Transmission]
    susceptibility: Expression
    infectivity: Expression
    transmissibility: Expression
    enabled: Expression
    state: StateAccessor
    scope: Scope | None = Field(default=None, repr=False)
    pos: SourcePosition | None = Field(default=None, repr=False)

    @classmethod
    def make(cls, node: PTNode, parent_scope: Scope) -> Contagion:
        name = node.field("name").text
        scope = Scope(name=name, parent=parent_scope)

        children = defaultdict(list)
        for child in node.fields("body"):
            children[child.type].append(parse(child, scope))

        state = StateAccessor(scope=scope)
        scope.define("state", state)

        state_type = children["contagion_state_type"]
        with err_desc("State type must defined once for each contagion.", node.pos):
            assert len(state_type) == 1
        state_type = state_type[0]

        transitions = [t for ts in children["transitions"] for t in ts]
        transmissions = [t for ts in children["transmissions"] for t in ts]
        contagion_functions = children["contagion_function"]

        susceptibility = [f for k, f in contagion_functions if k == "susceptibility"]
        with err_desc("Susceptibility must defined once for each contagion.", node.pos):
            assert len(susceptibility) == 1
        susceptibility = susceptibility[0]
        susceptibility = if_fn_make_ref(susceptibility, scope)

        infectivity = [f for k, f in contagion_functions if k == "infectivity"]
        with err_desc("Infectivity must defined once for each contagion.", node.pos):
            assert len(infectivity) == 1
        infectivity = infectivity[0]
        infectivity = if_fn_make_ref(infectivity, scope)

        transmissibility = [
            f for k, f in contagion_functions if k == "transmissibility"
        ]
        with err_desc(
            "Transmissibility must defined once for each contagion.", node.pos
        ):
            assert len(transmissibility) == 1
        transmissibility = transmissibility[0]
        transmissibility = if_fn_make_ref(transmissibility, scope)

        enabled = [f for k, f in contagion_functions if k == "enabled"]
        with err_desc("Enabled (edge) must defined once for each contagion.", node.pos):
            assert len(enabled) == 1
        enabled = enabled[0]
        enabled = if_fn_make_ref(enabled, scope)

        obj = cls(
            name=name,
            state_type=state_type,
            transitions=transitions,
            transmissions=transmissions,
            susceptibility=susceptibility,
            infectivity=infectivity,
            transmissibility=transmissibility,
            enabled=enabled,
            state=state,
            scope=scope,
            pos=node.pos,
        )
        parent_scope.define(name, obj)
        return obj


register_parser("contagion", Contagion.make)


class NodeSet(BaseModel):
    name: str
    pos: SourcePosition | None = Field(default=None, repr=False)

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> list[NodeSet]:
        sets: list[NodeSet] = []
        for name in node.fields("name"):
            obj = cls(name=name.text, pos=name.pos)
            scope.define(name.text, obj)
            sets.append(obj)
        return sets


register_parser("nodeset", NodeSet.make)


class EdgeSet(BaseModel):
    name: str
    pos: SourcePosition | None = Field(default=None, repr=False)

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> list[EdgeSet]:
        sets: list[EdgeSet] = []
        for name in node.fields("name"):
            obj = cls(name=name.text, pos=name.pos)
            scope.define(name.text, obj)
            sets.append(obj)
        return sets


register_parser("edgeset", EdgeSet.make)


class Param(BaseModel):
    name: str
    type: Reference
    scope: Scope | None = Field(default=None, repr=False)
    pos: SourcePosition | None = Field(default=None, repr=False)

    def model_post_init(self, _: Any) -> None:
        register_instance(self)

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> Param:
        name = node.field("name").text
        type = parse(node.field("type"), scope)
        obj = cls(name=name, type=type, scope=None, pos=node.pos)
        scope.define(name, obj)
        return obj

    @classmethod
    def link_tables(cls, node_table: NodeTable, edge_table: EdgeTable):
        obj: Param
        for obj in get_instances(cls):
            match obj.type.value:
                case BuiltinType(name="node"):
                    obj.scope = node_table.scope
                case BuiltinType(name="edge"):
                    obj.scope = edge_table.scope


register_parser("function_param", Param.make)


class Function(BaseModel):
    name: str
    params: list[Param]
    rtype: Reference | None
    body: list[Statement]
    scope: Scope | None = Field(default=None, repr=False)
    pos: SourcePosition | None = Field(default=None, repr=False)

    def model_post_init(self, _: Any) -> None:
        register_instance(self)

    @classmethod
    def make(cls, node: PTNode, parent_scope: Scope) -> Function:
        name = node.field("name").text
        scope = Scope(name=name, parent=parent_scope)

        params = []
        for child in node.field("params").named_children:
            params.append(parse(child, scope))

        rtype = node.maybe_field("rtype")
        if rtype is not None:
            rtype = parse(rtype, scope)

        body = []
        for child in node.field("body").named_children:
            stmt = parse(child, scope)
            body.append(stmt)

        obj = cls(
            name=name, params=params, rtype=rtype, body=body, scope=scope, pos=node.pos
        )
        parent_scope.define(name, obj)
        return obj

    @classmethod
    def make_inline_expression(cls, node: PTNode, parent_scope: Scope) -> Function:
        name = f"inline_expression_function_{node.pos.line}_{node.pos.col}"
        parent_scope = parent_scope.root()
        scope = Scope(name=name, parent=parent_scope)

        param_name = node.field("name").text
        param_type = parse(node.field("type"), scope)
        param = Param(name=param_name, type=param_type, scope=scope, pos=node.pos)
        scope.define(param_name, param)
        params = [param]

        rtype = node.maybe_field("rtype")
        if rtype is not None:
            rtype = parse(rtype, scope)

        expression = parse(node.field("expression"), scope)
        return_stmt = ReturnStatement(expression=expression, pos=node.pos)
        body = [return_stmt]
        body = cast(list[Statement], body)

        obj = cls(
            name=name, params=params, rtype=rtype, body=body, scope=scope, pos=node.pos
        )
        parent_scope.define(name, obj)
        return obj

    @classmethod
    def make_inline_update(cls, node: PTNode, parent_scope: Scope) -> Function:
        name = f"inline_update_function_{node.pos.line}_{node.pos.col}"
        parent_scope = parent_scope.root()
        scope = Scope(name=name, parent=parent_scope)

        param_name = node.field("name").text
        param_type = parse(node.field("type"), scope)
        param = Param(name=param_name, type=param_type, scope=scope, pos=node.pos)
        scope.define(param_name, param)
        params = [param]

        body = []
        for child in node.fields("stmt"):
            body.append(parse(child, scope))

        obj = cls(
            name=name, params=params, rtype=None, body=body, scope=scope, pos=node.pos
        )
        parent_scope.define(name, obj)
        return obj

    def variables(self) -> list[Variable]:
        assert self.scope is not None

        vars = [v for v in self.scope.names.values() if isinstance(v, Variable)]
        return vars


register_parser("function", Function.make)
register_parser("inline_expression_function", Function.make_inline_expression)
register_parser("inline_update_function", Function.make_inline_update)


class UnaryExpression(BaseModel):
    operator: str
    argument: Expression

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> UnaryExpression:
        operator = node.field("operator").text
        argument = parse(node.field("argument"), scope)
        return cls(operator=operator, argument=argument)


register_parser("unary_expression", UnaryExpression.make)


class BinaryExpression(BaseModel):
    left: Expression
    operator: str
    right: Expression

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> BinaryExpression:
        left = parse(node.field("left"), scope)
        operator = node.field("operator").text
        right = parse(node.field("right"), scope)
        return cls(left=left, operator=operator, right=right)


register_parser("binary_expression", BinaryExpression.make)


class ParenthesizedExpression(BaseModel):
    expression: Expression

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> ParenthesizedExpression:
        expression = parse(node.field("expression"), scope)
        return cls(expression=expression)


register_parser("parenthesized_expression", ParenthesizedExpression.make)


class FunctionCall(BaseModel):
    function: Reference
    args: list[Expression]
    parent_scope: str
    pos: SourcePosition | None = Field(default=None, repr=False)

    def model_post_init(self, _: Any) -> None:
        register_instance(self)

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> FunctionCall:
        function = parse(node.field("function"), scope)
        args = [parse(c, scope) for c in node.fields("arg")]
        obj = cls(function=function, args=args, parent_scope=scope.name, pos=node.pos)
        return obj


register_parser("function_call", FunctionCall.make)


class Variable(BaseModel):
    name: str
    type: Reference
    init: Expression
    scope: Scope | None = Field(default=None, repr=False)
    pos: SourcePosition | None = Field(default=None, repr=False)

    def model_post_init(self, _: Any) -> None:
        register_instance(self)

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> Variable:
        name = node.field("name").text
        type = parse(node.field("type"), scope)
        init = parse(node.field("init"), scope)
        obj = cls(name=name, type=type, init=init, scope=None, pos=node.pos)
        scope.define(name, obj)
        return obj

    @classmethod
    def link_tables(cls, node_table: NodeTable, edge_table: EdgeTable):
        obj: Variable
        for obj in get_instances(cls):
            match obj.type.value:
                case BuiltinType(name="node"):
                    obj.scope = node_table.scope
                case BuiltinType(name="edge"):
                    obj.scope = edge_table.scope


register_parser("variable", Variable.make)


class PassStatement(BaseModel):
    pos: SourcePosition | None = Field(default=None, repr=False)

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> PassStatement:
        scope = scope
        return cls(pos=node.pos)


register_parser("pass_statement", PassStatement.make)


class ReturnStatement(BaseModel):
    expression: Expression
    pos: SourcePosition | None = Field(default=None, repr=False)

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> ReturnStatement:
        expression = node.named_children[0]
        expression = parse(expression, scope)
        return cls(expression=expression, pos=node.pos)


register_parser("return_statement", ReturnStatement.make)


class ElifSection(BaseModel):
    condition: Expression
    body: list[Statement]

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> ElifSection:
        condition = parse(node.field("condition"), scope)
        body = []
        for child in node.fields("body"):
            body.append(parse(child, scope))
        return cls(condition=condition, body=body)


register_parser("elif_section", ElifSection.make)


class ElseSection(BaseModel):
    body: list[Statement]

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> ElseSection:
        body = []
        for child in node.fields("body"):
            body.append(parse(child, scope))
        return cls(body=body)


register_parser("else_section", ElseSection.make)


class IfStatement(BaseModel):
    condition: Expression
    body: list[Statement]
    elifs: list[ElifSection]
    else_: ElseSection | None
    pos: SourcePosition | None = Field(default=None, repr=False)

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> IfStatement:
        condition = parse(node.field("condition"), scope)
        body = []
        for child in node.fields("body"):
            body.append(parse(child, scope))
        elifs = []
        for child in node.fields("elif"):
            elifs.append(parse(child, scope))
        else_ = node.maybe_field("else")
        if else_ is not None:
            else_ = parse(else_, scope)

        obj = cls(
            condition=condition, body=body, elifs=elifs, else_=else_, pos=node.pos
        )
        return obj


register_parser("if_statement", IfStatement.make)


class CaseSection(BaseModel):
    match: Expression
    body: list[Statement]

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> CaseSection:
        match = parse(node.field("match"), scope)
        body = []
        for child in node.fields("body"):
            body.append(parse(child, scope))
        return cls(match=match, body=body)


register_parser("case_section", CaseSection.make)


class DefaultSection(BaseModel):
    body: list[Statement]

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> DefaultSection:
        body = []
        for child in node.fields("body"):
            body.append(parse(child, scope))
        return cls(body=body)


register_parser("default_section", DefaultSection.make)


class SwitchStatement(BaseModel):
    condition: Expression
    cases: list[CaseSection]
    default: DefaultSection | None
    pos: SourcePosition | None = Field(default=None, repr=False)

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> SwitchStatement:
        condition = parse(node.field("condition"), scope)
        cases = []
        for case in node.fields("case"):
            cases.append(parse(case, scope))
        default = node.maybe_field("default")
        if default is not None:
            default = parse(default, scope)

        obj = cls(condition=condition, cases=cases, default=default, pos=node.pos)
        return obj


register_parser("switch_statement", SwitchStatement.make)


class WhileLoop(BaseModel):
    condition: Expression
    body: list[Statement]
    pos: SourcePosition | None = Field(default=None, repr=False)

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> WhileLoop:
        condition = parse(node.field("condition"), scope)
        body = []
        for child in node.fields("body"):
            body.append(parse(child, scope))
        return cls(condition=condition, body=body, pos=node.pos)


register_parser("while_loop", WhileLoop.make)


class CallStatement(BaseModel):
    call: FunctionCall
    pos: SourcePosition | None = Field(default=None, repr=False)

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> CallStatement:
        call = parse(node.named_children[0], scope)
        return cls(call=call, pos=node.pos)


register_parser("call_statement", CallStatement.make)


class UpdateStatement(BaseModel):
    left: Reference
    operator: str
    right: Expression
    pos: SourcePosition | None = Field(default=None, repr=False)

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> UpdateStatement:
        left = parse(node.field("left"), scope)
        operator = node.field("operator").text
        right = parse(node.field("right"), scope)
        obj = cls(left=left, operator=operator, right=right, pos=node.pos)
        return obj


register_parser("update_statement", UpdateStatement.make)


class PrintStatement(BaseModel):
    args: list[str | Expression]
    sep: str
    end: str
    pos: SourcePosition | None = Field(default=None, repr=False)

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> PrintStatement:
        args = []
        for child in node.named_children:
            args.append(parse(child, scope))
        sep = '" "'
        end = r'"\n"'
        return cls(args=args, sep=sep, end=end, pos=node.pos)


register_parser("print_statement", PrintStatement.make)


class SelectStatement(BaseModel):
    name: str
    set: Reference
    function: Reference
    parent_scope: str
    pos: SourcePosition | None = Field(default=None, repr=False)

    def model_post_init(self, _: Any) -> None:
        register_instance(self)

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> SelectStatement:
        set = parse(node.field("set"), scope)
        function = parse(node.field("function"), scope)
        function = if_fn_make_ref(function, scope)
        name = f"select_{set.name}_{function.name}_{node.pos.line}_{node.pos.col}"
        obj = cls(
            name=name, set=set, function=function, parent_scope=scope.name, pos=node.pos
        )
        return obj


register_parser("select_statement", SelectStatement.make)


class SampleStatement(BaseModel):
    name: str
    set: Reference
    parent: Reference
    amount: Expression
    is_absolute: bool
    parent_scope: str
    pos: SourcePosition | None = Field(default=None, repr=False)

    def model_post_init(self, _: Any) -> None:
        register_instance(self)

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> SampleStatement:
        set = parse(node.field("set"), scope)
        parent = parse(node.field("parent"), scope)
        amount = parse(node.field("amount"), scope)
        name = f"sample_{set.name}_{parent.name}_{node.pos.line}_{node.pos.col}"
        is_absolute = node.field("type").text == "ABSOLUTE"
        obj = cls(
            name=name,
            set=set,
            parent=parent,
            amount=amount,
            is_absolute=is_absolute,
            parent_scope=scope.name,
            pos=node.pos,
        )
        return obj


register_parser("sample_statement", SampleStatement.make)


class ApplyStatement(BaseModel):
    name: str
    set: Reference
    function: Reference
    parent_scope: str
    pos: SourcePosition | None = Field(default=None, repr=False)

    def model_post_init(self, _: Any) -> None:
        register_instance(self)

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> ApplyStatement:
        set = parse(node.field("set"), scope)
        function = parse(node.field("function"), scope)
        function = if_fn_make_ref(function, scope)
        name = f"apply_{set.name}_{function.name}_{node.pos.line}_{node.pos.col}"
        obj = cls(
            name=name, set=set, function=function, parent_scope=scope.name, pos=node.pos
        )

        return obj


register_parser("apply_statement", ApplyStatement.make)

ReduceOperatorType = Literal["+", "*"]


class ReduceStatement(BaseModel):
    name: str
    outvar: Reference
    set: Reference
    function: Reference
    operator: ReduceOperatorType
    parent_scope: str
    pos: SourcePosition | None = Field(default=None, repr=False)

    def model_post_init(self, _: Any) -> None:
        register_instance(self)

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> ReduceStatement:
        outvar = parse(node.field("outvar"), scope)
        set = parse(node.field("set"), scope)
        function = parse(node.field("function"), scope)
        function = if_fn_make_ref(function, scope)
        operator = node.field("operator").text
        operator = cast(ReduceOperatorType, operator)
        name = f"reduce_{set.name}_{function.name}_{node.pos.line}_{node.pos.col}"
        obj = cls(
            name=name,
            outvar=outvar,
            set=set,
            function=function,
            operator=operator,
            parent_scope=scope.name,
            pos=node.pos,
        )
        return obj


register_parser("reduce_statement", ReduceStatement.make)


Expression = (
    int
    | float
    | bool
    | UnaryExpression
    | BinaryExpression
    | ParenthesizedExpression
    | Reference
    | TemplateVariable
    | FunctionCall
)

SerialStatement = (
    PassStatement
    | ReturnStatement
    | IfStatement
    | SwitchStatement
    | WhileLoop
    | CallStatement
    | UpdateStatement
    | PrintStatement
    | Variable
)

ParallelStatement = SelectStatement | SampleStatement | ApplyStatement | ReduceStatement

Statement = SerialStatement | ParallelStatement

# References which can be assigned to
LValueRef = (
    Global
    | Param
    | Variable
    | tuple[Param | Variable, NodeField | EdgeField]
    | tuple[Param | Variable, Contagion, StateAccessor]
    | tuple[Param | Variable, SourceNodeAccessor | TargetNodeAccessor, NodeField]
    | tuple[
        Param | Variable,
        SourceNodeAccessor | TargetNodeAccessor,
        Contagion,
        StateAccessor,
    ]
)

# References which can be evaluated to get a value
RValueRef = (
    BuiltinGlobal
    | EnumConstant
    | Global
    | Param
    | Variable
    | NodeSet
    | BuiltinNodeset
    | EdgeSet
    | BuiltinEdgeset
    | tuple[Param | Variable, NodeField | EdgeField]
    | tuple[Param | Variable, Contagion, StateAccessor]
    | tuple[Param | Variable, SourceNodeAccessor | TargetNodeAccessor]
    | tuple[Param | Variable, SourceNodeAccessor | TargetNodeAccessor, NodeField]
    | tuple[
        Param | Variable,
        SourceNodeAccessor | TargetNodeAccessor,
        Contagion,
        StateAccessor,
    ]
)

# References which are valid types
TValueRef = BuiltinType | EnumType

# Things that can be called
CValueRef = BuiltinFunction | Function | Distribution


SIGNED_INT_TYPES = {"int", "i8", "i16", "i32", "i64"}
UNSIGNED_INT_TYPES = {"uint", "u8", "u16", "u32", "u64"}

FLOAT_TYPES = {"float", "f32", "f64"}

INT_TYPES = SIGNED_INT_TYPES | UNSIGNED_INT_TYPES | {"size"}
SCALAR_TYPES = INT_TYPES | FLOAT_TYPES | {"bool"}

TABLE_TYPES = {"node", "edge"}
SET_TYPES = {"nodeset", "edgeset"}

BUILTIN_TYPES = SCALAR_TYPES | TABLE_TYPES | SET_TYPES


def add_builtins(scope: Scope):
    for type in BUILTIN_TYPES:
        scope.define(type, BuiltinType(name=type))

    scope.define(
        "NUM_TICKS", BuiltinGlobal(name="NUM_TICKS", type="int", is_config=True)
    )
    scope.define(
        "CUR_TICK", BuiltinGlobal(name="CUR_TICK", type="int", is_config=False)
    )
    scope.define(
        "NUM_NODES", BuiltinGlobal(name="NUM_NODES", type="size", is_config=False)
    )
    scope.define(
        "NUM_EDGES", BuiltinGlobal(name="NUM_EDGES", type="size", is_config=False)
    )

    scope.define("ALL_NODES", BuiltinNodeset(name="ALL_NODES"))
    scope.define("ALL_EDGES", BuiltinEdgeset(name="ALL_EDGES"))

    scope.define(
        "len",
        BuiltinFunction(
            name="len",
            params=[BuiltinParam(name="x", type="nodeset|edgeset")],
            rtype="size",
        ),
    )
    scope.define(
        "abs",
        BuiltinFunction(
            name="abs",
            params=[BuiltinParam(name="x", type="float")],
            rtype="float",
        ),
    )
    scope.define(
        "min",
        BuiltinFunction(
            name="min",
            params=[BuiltinParam(name="x", type="float"), BuiltinParam(name="y", type="float")],
            rtype="float",
        ),
    )
    scope.define(
        "max",
        BuiltinFunction(
            name="max",
            params=[BuiltinParam(name="x", type="float"), BuiltinParam(name="y", type="float")],
            rtype="float",
        ),
    )
    scope.define(
        "exp",
        BuiltinFunction(
            name="exp",
            params=[BuiltinParam(name="x", type="float")],
            rtype="float",
        ),
    )
    scope.define(
        "exp2",
        BuiltinFunction(
            name="exp2",
            params=[BuiltinParam(name="x", type="float")],
            rtype="float",
        ),
    )
    scope.define(
        "log",
        BuiltinFunction(
            name="log",
            params=[BuiltinParam(name="x", type="float")],
            rtype="float",
        ),
    )
    scope.define(
        "log2",
        BuiltinFunction(
            name="log2",
            params=[BuiltinParam(name="x", type="float")],
            rtype="float",
        ),
    )
    scope.define(
        "pow",
        BuiltinFunction(
            name="pow",
            params=[BuiltinParam(name="base", type="float"), BuiltinParam(name="exp", type="float")],
            rtype="float",
        ),
    )
    scope.define(
        "sin",
        BuiltinFunction(
            name="sin",
            params=[BuiltinParam(name="x", type="float")],
            rtype="float",
        ),
    )
    scope.define(
        "cos",
        BuiltinFunction(
            name="cos",
            params=[BuiltinParam(name="x", type="float")],
            rtype="float",
        ),
    )


class Source(BaseModel):
    module: str
    enums: list[EnumType]
    globals: list[Global]
    node_table: NodeTable
    edge_table: EdgeTable
    normal_dists: list[NormalDist]
    uniform_dists: list[UniformDist]
    discrete_dists: list[DiscreteDist]
    contagions: list[Contagion]
    nodesets: list[NodeSet]
    edgesets: list[EdgeSet]
    functions: list[Function]
    function_calls: list[FunctionCall]
    select_statements: list[SelectStatement]
    sample_statements: list[SampleStatement]
    apply_statements: list[ApplyStatement]
    reduce_statements: list[ReduceStatement]
    intervene: Function
    template_variables: list[TemplateVariable]
    scope: Scope | None = Field(default=None, repr=False)
    pos: SourcePosition | None = Field(default=None, repr=False)

    @classmethod
    def make(cls, module: str, node: PTNode) -> Source:
        scope = Scope(name="source")

        children = defaultdict(list)
        clear_instances()

        for child in node.named_children:
            children[child.type].append(parse(child, scope))

        enums = children["enum"]
        globals = children["global"]

        node_tables = children["node"]
        with err_desc("One and only one node table must be defined", node.pos):
            assert len(node_tables) == 1
        node_table = node_tables[0]

        edge_tables = children["edge"]
        with err_desc("One and only one edge table must be defined", node.pos):
            assert len(edge_tables) == 1
        edge_table = edge_tables[0]

        distributions = [d for ds in children["distributions"] for d in ds]
        normal_dists = [d for d in distributions if isinstance(d, NormalDist)]
        uniform_dists = [d for d in distributions if isinstance(d, UniformDist)]
        discrete_dists = [d for d in distributions if isinstance(d, DiscreteDist)]

        contagions = children["contagion"]

        nodesets = [s for ss in children["nodeset"] for s in ss]
        edgesets = [s for ss in children["edgeset"] for s in ss]

        functions: list[Function] = get_instances(Function)

        intervenes = [f for f in functions if f.name == "intervene"]
        with err_desc("One and only one intervene function must be defined", node.pos):
            assert len(intervenes) == 1
        intervene = intervenes[0]

        function_calls: list[FunctionCall] = get_instances(FunctionCall)
        select_statements: list[SelectStatement] = get_instances(SelectStatement)
        sample_statements: list[SampleStatement] = get_instances(SampleStatement)
        apply_statements: list[ApplyStatement] = get_instances(ApplyStatement)
        reduce_statements: list[ReduceStatement] = get_instances(ReduceStatement)

        add_builtins(scope)

        node_table.link_contagions(contagions)
        edge_table.link_contagions(contagions)
        edge_table.link_node_table(node_table)
        Param.link_tables(node_table, edge_table)
        Variable.link_tables(node_table, edge_table)

        template_variables: list[TemplateVariable] = get_instances(TemplateVariable)

        # rich.print(scope.rich_tree())

        # At this point all references should be resolve without errors
        ref: Reference
        for ref in get_instances(Reference):
            _ = ref.value

        return cls(
            module=module,
            enums=enums,
            globals=globals,
            node_table=node_table,
            edge_table=edge_table,
            normal_dists=normal_dists,
            uniform_dists=uniform_dists,
            discrete_dists=discrete_dists,
            contagions=contagions,
            nodesets=nodesets,
            edgesets=edgesets,
            functions=functions,
            function_calls=function_calls,
            select_statements=select_statements,
            sample_statements=sample_statements,
            apply_statements=apply_statements,
            reduce_statements=reduce_statements,
            intervene=intervene,
            template_variables=template_variables,
            scope=scope,
            pos=node.pos,
        )


def mk_ast(filename: Path, node: PTNode) -> Source:
    """Make AST from parse tree."""
    module = filename.stem.replace(".", "_")
    try:
        return Source.make(module, node)
    except EslError:
        raise
    except Exception as e:
        estr = f"{e.__class__.__name__}: {e!s}"
        raise EslError(
            "AST Error", f"Failed to parse '{node.type}': ({estr})", node.pos
        )


@click.command()
@simulation_file_option
def print_ast(simulation_file: Path):
    """Print the AST."""
    file_bytes = simulation_file.read_bytes()
    try:
        pt = mk_pt(str(simulation_file), file_bytes)
        ast = mk_ast(simulation_file, pt)
        rich.print(ast)
    except RichException as e:
        e.rich_print()
        raise SystemExit(1)
