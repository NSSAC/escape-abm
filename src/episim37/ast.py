"""Abstract syntax tree."""

from __future__ import annotations

from pathlib import Path
from collections import defaultdict
from typing import Any, ClassVar, Self, Callable, Literal, cast

import click
import pydantic
import rich
import rich.markup
from rich.tree import Tree
from rich.pretty import Pretty
from pydantic import BaseModel, Field, model_validator

from .misc import EslError, SourcePosition, validation_error_str
from .parse_tree import mk_pt, PTNode, ParseTreeConstructionError
from .click_helpers import simulation_file_option


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

    def visit(self):
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
    except pydantic.ValidationError as e:
        e_str = validation_error_str(e)
        raise EslError("AST Error", f"Failed to parse '{node.type}'\n{e_str}", node.pos)
    except Exception as e:
        raise EslError("AST Error", f"Failed to parse '{node.type}': {e!s}", node.pos)


def register_parser(type: str, parser: NodeParser):
    if type in _NODE_PARSER:
        raise ValueError(f"Parser for '{type}' is already defined")
    _NODE_PARSER[type] = parser


register_parser("integer", lambda node, _: int(node.text))
register_parser("float", lambda node, _: float(node.text))
register_parser("boolean", lambda node, _: node.text == "True")
register_parser("string", lambda node, _: node.text)


class Reference(BaseModel):
    instances: ClassVar[list[Reference]] = []

    names: list[str]
    scope: Scope = Field(repr=False)
    pos: SourcePosition | None = Field(default=None, repr=False)
    value: Any | None = Field(default=None, repr=False)

    def model_post_init(self, _: Any) -> None:
        Reference.instances.append(self)

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> Reference:
        names = [s.text for s in node.named_children]
        obj = cls(names=names, scope=scope, pos=node.pos)
        return obj

    @property
    def name(self) -> str:
        return ".".join(self.names)

    def resolve(self) -> Any:
        if self.value is not None:
            return self.value

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

        self.value = value
        return self.value


register_parser("reference", Reference.make)


class TemplateVariable(BaseModel):
    pos: SourcePosition | None = Field(default=None, repr=False)

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
    is_device: bool


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

    @model_validator(mode="after")
    def check_static_save_conflict(self) -> Self:
        if self.save_to_output and self.is_static:
            raise ValueError("Static field can't be saved to output")
        return self


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
        if len(key) != 1:
            raise ValueError("One and only one node key field must be specified.")
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

        if save_to_output and is_static:
            raise EslError(
                "Saving static field", "Static field can't be saved in output", node.pos
            )

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

    @model_validator(mode="after")
    def check_static_save_conflict(self) -> Self:
        if self.save_to_output and self.is_static:
            raise ValueError("Static field can't be saved to output")
        return self


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
        if len(target_key) != 1:
            raise ValueError(
                "One and only one target node key field must be specified."
            )
        target_key = target_key[0]

        source_key = [f for f in fields if f.is_source_node_key]
        if len(source_key) != 1:
            raise ValueError(
                "One and only one source node key field must be specified."
            )
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


class Transition(BaseModel):
    entry: Reference
    exit: Reference
    pfunc: Expression | Function | None
    dwell: Expression | Function
    pos: SourcePosition | None = Field(default=None, repr=False)

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> Transition:
        entry = parse(node.field("entry"), scope)
        exit = parse(node.field("exit"), scope)
        pfunc = node.maybe_field("pfunc")
        if pfunc is not None:
            pfunc = parse(pfunc, scope)
        dwell = parse(node.field("dwell"), scope)
        return cls(entry=entry, exit=exit, pfunc=pfunc, dwell=dwell, pos=node.pos)


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
    susceptibility: Expression | Function
    infectivity: Expression | Function
    transmissibility: Expression | Function
    enabled: Expression | Function
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
        if len(state_type) != 1:
            raise ValueError("State type must defined once for each contagion")
        state_type = state_type[0]

        transitions = [t for ts in children["transitions"] for t in ts]
        transmissions = [t for ts in children["transmissions"] for t in ts]

        susceptibility = [
            f for k, f in children["contagion_function"] if k == "susceptibility"
        ]
        if len(susceptibility) != 1:
            raise ValueError("Susceptibility must defined once for each contagion")
        susceptibility = susceptibility[0]

        infectivity = [
            f for k, f in children["contagion_function"] if k == "infectivity"
        ]
        if len(infectivity) != 1:
            raise ValueError("Infectivity must defined once for each contagion")
        infectivity = infectivity[0]

        transmissibility = [
            f for k, f in children["contagion_function"] if k == "transmissibility"
        ]
        if len(transmissibility) != 1:
            raise ValueError("Transmissibility must defined once for each contagion")
        transmissibility = transmissibility[0]

        enabled = [f for k, f in children["contagion_function"] if k == "enabled"]
        if len(enabled) != 1:
            raise ValueError("Enabled (edge) must defined once for each contagion")
        enabled = enabled[0]

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
    instances: ClassVar[list[Param]] = []

    name: str
    type: Reference
    scope: Scope | None = Field(default=None, repr=False)
    pos: SourcePosition | None = Field(default=None, repr=False)

    def model_post_init(self, _: Any) -> None:
        Param.instances.append(self)

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> Param:
        name = node.field("name").text
        type = parse(node.field("type"), scope)
        obj = cls(name=name, type=type, scope=None, pos=node.pos)
        scope.define(name, obj)
        return obj

    @classmethod
    def link_tables(cls, node_table: NodeTable, edge_table: EdgeTable):
        for obj in cls.instances:
            match obj.type.resolve():
                case BuiltinType(name="node"):
                    obj.scope = node_table.scope
                case BuiltinType(name="edge"):
                    obj.scope = edge_table.scope


register_parser("function_param", Param.make)


class Function(BaseModel):
    instances: ClassVar[list[Function]] = []

    name: str
    params: list[Param]
    rtype: Reference | None
    body: list[Statement]
    scope: Scope | None = Field(default=None, repr=False)
    pos: SourcePosition | None = Field(default=None, repr=False)

    def model_post_init(self, _: Any) -> None:
        Function.instances.append(self)

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
    instances: ClassVar[list[FunctionCall]] = []

    function: Reference
    args: list[Expression]
    parent_scope: str
    pos: SourcePosition | None = Field(default=None, repr=False)

    def model_post_init(self, _: Any) -> None:
        FunctionCall.instances.append(self)

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> FunctionCall:
        function = parse(node.field("function"), scope)
        args = [parse(c, scope) for c in node.fields("arg")]
        obj = cls(function=function, args=args, parent_scope=scope.name, pos=node.pos)
        return obj


register_parser("function_call", FunctionCall.make)


class Variable(BaseModel):
    instances: ClassVar[list[Variable]] = []

    name: str
    type: Reference
    init: Expression
    scope: Scope | None = Field(default=None, repr=False)
    pos: SourcePosition | None = Field(default=None, repr=False)

    def model_post_init(self, _: Any) -> None:
        Variable.instances.append(self)

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
        for obj in cls.instances:
            match obj.type.resolve():
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
    instances: ClassVar[list[SelectStatement]] = []

    name: str
    set: Reference
    function: Reference | Function
    parent_scope: str
    pos: SourcePosition | None = Field(default=None, repr=False)

    def model_post_init(self, _: Any) -> None:
        SelectStatement.instances.append(self)

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> SelectStatement:
        set = parse(node.field("set"), scope)
        function = parse(node.field("function"), scope)
        name = f"select_{set.name}_{function.name}_{node.pos.line}_{node.pos.col}"
        obj = cls(
            name=name, set=set, function=function, parent_scope=scope.name, pos=node.pos
        )
        return obj


register_parser("select_statement", SelectStatement.make)


class SampleStatement(BaseModel):
    instances: ClassVar[list[SampleStatement]] = []

    name: str
    set: Reference
    parent: Reference
    amount: Expression
    is_absolute: bool
    parent_scope: str
    pos: SourcePosition | None = Field(default=None, repr=False)

    def model_post_init(self, _: Any) -> None:
        SampleStatement.instances.append(self)

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
    instances: ClassVar[list[ApplyStatement]] = []

    name: str
    set: Reference
    function: Reference | Function
    parent_scope: str
    pos: SourcePosition | None = Field(default=None, repr=False)

    def model_post_init(self, _: Any) -> None:
        ApplyStatement.instances.append(self)

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> ApplyStatement:
        set = parse(node.field("set"), scope)
        function: Reference | Function
        function = parse(node.field("function"), scope)
        name = f"apply_{set.name}_{function.name}_{node.pos.line}_{node.pos.col}"
        obj = cls(
            name=name, set=set, function=function, parent_scope=scope.name, pos=node.pos
        )

        return obj


register_parser("apply_statement", ApplyStatement.make)

ReduceOperatorType = Literal["+", "*"]


class ReduceStatement(BaseModel):
    instances: ClassVar[list[ReduceStatement]] = []

    name: str
    outvar: Reference
    set: Reference
    function: Reference | Function
    operator: ReduceOperatorType
    parent_scope: str
    pos: SourcePosition | None = Field(default=None, repr=False)

    def model_post_init(self, _: Any) -> None:
        ReduceStatement.instances.append(self)

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> ReduceStatement:
        outvar = parse(node.field("outvar"), scope)
        set = parse(node.field("set"), scope)
        function: Reference | Function
        function = parse(node.field("function"), scope)
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

HostStatement = (
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

DeviceStatement = SelectStatement | SampleStatement | ApplyStatement | ReduceStatement

Statement = HostStatement | DeviceStatement

StatementParts = Statement | ElifSection | ElseSection | CaseSection | DefaultSection


Referable = (
    BuiltinType
    | BuiltinFunction
    | BuiltinGlobal
    | BuiltinNodeset
    | BuiltinEdgeset
    | EnumConstant
    | EnumType
    | Global
    | Param
    | Variable
    | Function
    | Distribution
    | NodeSet
    | EdgeSet
    | tuple[Param | Variable, NodeField | EdgeField]
    | tuple[Param | Variable, Contagion, StateAccessor]
    | tuple[Param | Variable, SourceNodeAccessor | TargetNodeAccessor]
    | tuple[Param | Variable, SourceNodeAccessor | TargetNodeAccessor, NodeField]
)

EslCallable = BuiltinFunction | Function | Distribution

Updateable = (
    Global
    | Param
    | Variable
    | tuple[Param | Variable, NodeField | EdgeField]
    | tuple[Param | Variable, Contagion, StateAccessor]
    | tuple[Param | Variable, SourceNodeAccessor | TargetNodeAccessor, NodeField]
)


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
            params=[BuiltinParam(name="x", type="nodeset_or_edgeset")],
            rtype="size",
            is_device=False,
        ),
    )


class Source(BaseModel):
    module: str
    enums: list[EnumType]
    globals: list[Global]
    node_table: NodeTable
    edge_table: EdgeTable
    distributions: list[Distribution]
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
    scope: Scope | None = Field(default=None, repr=False)
    pos: SourcePosition | None = Field(default=None, repr=False)

    @classmethod
    def make(cls, module: str, node: PTNode) -> Source:
        scope = Scope(name="source")

        children = defaultdict(list)
        Function.instances.clear()
        FunctionCall.instances.clear()
        SelectStatement.instances.clear()
        SampleStatement.instances.clear()
        ApplyStatement.instances.clear()
        ReduceStatement.instances.clear()
        Param.instances.clear()
        Variable.instances.clear()
        Reference.instances.clear()

        for child in node.named_children:
            children[child.type].append(parse(child, scope))

        enums = children["enum"]
        globals = children["global"]

        node_tables = children["node"]
        if len(node_tables) != 1:
            raise ValueError("One and only one node table must be defined")
        node_table = node_tables[0]

        edge_tables = children["edge"]
        if len(edge_tables) != 1:
            raise ValueError("One and only one edge table must be defined")
        edge_table = edge_tables[0]

        distributions = [d for ds in children["distributions"] for d in ds]
        contagions = children["contagion"]

        nodesets = [s for ss in children["nodeset"] for s in ss]
        edgesets = [s for ss in children["edgeset"] for s in ss]

        functions = list(Function.instances)

        intervenes = [f for f in functions if f.name == "intervene"]
        if len(intervenes) != 1:
            raise ValueError("One and only one intervene function must be defined")
        intervene = intervenes[0]

        function_calls = list(FunctionCall.instances)
        select_statements = list(SelectStatement.instances)
        sample_statements = list(SampleStatement.instances)
        apply_statements = list(ApplyStatement.instances)
        reduce_statements = list(ReduceStatement.instances)

        add_builtins(scope)

        node_table.link_contagions(contagions)
        edge_table.link_contagions(contagions)
        edge_table.link_node_table(node_table)
        Param.link_tables(node_table, edge_table)
        Variable.link_tables(node_table, edge_table)

        # rich.print(scope.rich_tree())

        # At this point all references should be resolve without errors
        for ref in Reference.instances:
            ref.resolve()

        return cls(
            module=module,
            enums=enums,
            globals=globals,
            node_table=node_table,
            edge_table=edge_table,
            distributions=distributions,
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
            scope=scope,
            pos=node.pos,
        )


def mk_ast(filename: Path, node: PTNode) -> Source:
    """Make AST from parse tree."""
    module = filename.stem.replace(".", "_")
    try:
        source = Source.make(module, node)
    except EslError:
        raise
    except pydantic.ValidationError as e:
        e_str = validation_error_str(e)
        raise EslError("AST Error", f"Failed to parse '{node.type}'\n{e_str}", node.pos)
    except Exception as e:
        raise EslError("AST Error", f"Failed to parse '{node.type}'\n{e!s}", node.pos)
    return source


@click.command()
@simulation_file_option
def print_ast(simulation_file: Path):
    """Print the AST."""
    file_bytes = simulation_file.read_bytes()
    try:
        pt = mk_pt(str(simulation_file), file_bytes)
        ast = mk_ast(simulation_file, pt)
        rich.print(ast)
    except (ParseTreeConstructionError, EslError) as e:
        e.rich_print()
        raise SystemExit(1)