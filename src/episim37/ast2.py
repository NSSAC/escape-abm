"""Abstract syntax tree v2."""

from __future__ import annotations

from typing import Any

import attrs
import click

import rich
import rich.markup

from .parse_tree import mk_pt, ParseTreeConstructionError, SourcePosition
from . import ast1

Error = ast1.ASTConstructionError


@attrs.define
class Config:
    name: str
    type: ast1.BuiltinType
    default: int | float | bool
    pos: SourcePosition | None = attrs.field(default=None, repr=False, eq=False)

    @classmethod
    def make(cls, x: ast1.Config) -> Config:
        match x.type.resolve():
            case ast1.BuiltinType() as t:
                type = t
            case _:
                raise Error(
                    "Invalid type",
                    f"{x.type} is not a builtin type",
                    x.type.pos,
                )

        return cls(x.name, type, x.default, x.pos)


@attrs.define
class NodeField:
    name: str
    type: ast1.BuiltinType | ast1.EnumType
    is_node_key: bool
    is_static: bool
    pos: SourcePosition | None = attrs.field(default=None, repr=False, eq=False)

    @classmethod
    def make(cls, x: ast1.NodeField):
        match x.type.resolve():
            case ast1.BuiltinType() | ast1.EnumType() as t:
                type = t
            case _:
                raise Error(
                    "Invalid type",
                    f"{x.type} is not a built in or enumerated type",
                    x.type.pos,
                )
        return cls(x.name, type, x.is_node_key, x.is_static, x.pos)


@attrs.define
class NodeTable:
    fields: list[NodeField]
    key: NodeField
    scope: ast1.Scope | None = attrs.field(default=None, repr=False, eq=False)
    node: SourcePosition | None = attrs.field(default=None, repr=False, eq=False)

    @classmethod
    def make(cls, x: ast1.NodeTable) -> NodeTable:
        fields = [NodeField.make(n) for n in x.fields] 
        key = [f for f in fields if f.is_node_key]
        assert len(key) == 1
        key = key[0]

        obj = NodeTable(fields, key, x.scope, x.node)
        scope.define("node_table", obj, True, node)
        return obj

    def add_contagion_fields(self, c: Contagion):
        assert self.scope is not None
        assert c.scope is not None
        assert c.node is not None

        self.scope.define(c.name, c, True, c.node)

        field = NodeField(f"_{c.name}_state", c.state_type, False, False, c.node)
        c.scope.define("state", field, True, c.node)
        self.fields.append(field)

        field = NodeField(f"_{c.name}_next_state", c.state_type, False, False, c.node)
        c.scope.define("next_state", field, True, c.node)
        self.fields.append(field)

        field = NodeField(
            f"_{c.name}_dwell_time", BuiltinType("f32"), False, False, c.node
        )
        c.scope.define("dwell_time", field, True, c.node)
        self.fields.append(field)


@attrs.define
class EdgeField:
    name: str
    type: BuiltinType | EnumType
    is_target_node_key: bool
    is_source_node_key: bool
    is_static: bool
    node: PTNode | None = attrs.field(default=None, repr=False, eq=False)

    @classmethod
    def make(cls, node: PTNode, scope: Scope):
        name = node.field("name").text
        type_node = node.field("type")

        match scope.resolve(type_node.text, type_node):
            case [_, (BuiltinType() | EnumType()) as t]:
                type = t
            case _:
                raise Error(
                    "Invalid type",
                    f"{type_node.text} is not a built in or enumerated type",
                    type_node.pos,
                )

        annotations = [c.text for c in node.fields("annotation")]
        is_source_node_key = "source node key" in annotations
        is_target_node_key = "target node key" in annotations
        is_static = "static" in annotations or is_source_node_key or is_target_node_key

        obj = EdgeField(
            name, type, is_target_node_key, is_source_node_key, is_static, node
        )
        scope.define(name, obj, True, node)
        return obj


@attrs.define
class EdgeTable:
    fields: list[EdgeField]
    target_node_key: EdgeField
    source_node_key: EdgeField
    scope: Scope | None = attrs.field(default=None, repr=False, eq=False)
    node: PTNode | None = attrs.field(default=None, repr=False, eq=False)

    @classmethod
    def make(cls, node: PTNode, parent_scope: Scope) -> EdgeTable:
        scope = Scope(name="edge_table", parent=parent_scope)

        fields = []
        gen_field = EdgeField(
            "_source_node_idx", BuiltinType("u64"), False, False, True, node
        )
        scope.define(gen_field.name, gen_field, True, node)
        fields.append(gen_field)

        gen_field = EdgeField(
            "_target_node_idx", BuiltinType("u64"), False, False, True, node
        )
        scope.define(gen_field.name, gen_field, True, node)
        fields.append(gen_field)

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

        obj = EdgeTable(fields, target_key, source_key, scope, node)
        scope.define("edge_table", obj, True, node)
        return obj


@attrs.define
class ConstantDist:
    name: str
    v: int | float
    node: PTNode | None = attrs.field(default=None, repr=False, eq=False)

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> ConstantDist:
        name = node.field("name").text
        v = make_number(node.field("v"))
        obj = ConstantDist(name, v, node)
        scope.define(name, obj, True, node)
        return obj


@attrs.define
class DiscreteDist:
    name: str
    ps: list[int | float]
    vs: list[int | float]
    node: PTNode | None = attrs.field(default=None, repr=False, eq=False)

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> DiscreteDist:
        name = node.field("name").text
        ps = []
        vs = []
        for child in node.fields("pv"):
            p = make_number(child.field("p"))
            ps.append(p)

            v = make_number(child.field("v"))
            vs.append(v)

        obj = DiscreteDist(name, ps, vs, node)
        scope.define(name, obj, True, node)
        return obj


@attrs.define
class NormalDist:
    name: str
    mean: int | float
    std: int | float
    min: int | float
    max: int | float
    node: PTNode | None = attrs.field(default=None, repr=False, eq=False)

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> NormalDist:
        name = node.field("name").text
        mean = make_number(node.field("mean"))
        std = make_number(node.field("std"))

        min = node.maybe_field("min")
        min = -1e300 if min is None else make_number(min)

        max = node.maybe_field("max")
        max = 1e300 if max is None else make_number(max)

        obj = NormalDist(name, mean, std, min, max, node)
        scope.define(name, obj, True, node)
        return obj


@attrs.define
class UniformDist:
    name: str
    low: int | float
    high: int | float
    node: PTNode | None = attrs.field(default=None, repr=False, eq=False)

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> UniformDist:
        name = node.field("name").text
        low = make_number(node.field("low"))
        high = make_number(node.field("high"))
        obj = UniformDist(name, low, high, node)
        scope.define(name, obj, True, node)
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
class Contagion:
    name: str
    state_type: EnumType
    transitions: list
    transmissions: list
    susceptibility: Function
    infectivity: Function
    transmissibility: Function
    enabled: Function
    scope: Scope | None = attrs.field(default=None, repr=False, eq=False)
    node: PTNode | None = attrs.field(default=None, repr=False, eq=False)

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
                    match scope.resolve(child.field("type").text, child):
                        case _, EnumType() as t:
                            state_type = t
                        case _:
                            raise Error(
                                "Invalid type",
                                f"{child.field('type').text} is not a enumerated type",
                                child.field("type").pos,
                            )
                case "transitions":
                    pass
                case "transmissions":
                    pass
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

        obj = Contagion(
            name,
            state_type,
            transitions,
            transmissions,
            susceptibility,
            infectivity,
            transmissibility,
            enabled,
            scope,
            node,
        )
        scope.define(name, obj, True, node)
        return obj


@attrs.define
class PassStatement:
    node: PTNode | None = attrs.field(default=None, repr=False, eq=False)


@attrs.define
class ReturnStatement:
    expression: Expression
    node: PTNode | None = attrs.field(default=None, repr=False, eq=False)

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> ReturnStatement:
        expression = node.named_children[0]
        expression = parse_expression(expression, scope)
        return ReturnStatement(expression, node)


@attrs.define
class PrintStatement:
    args: list[str | Expression]
    sep: str
    end: str
    node: PTNode | None = attrs.field(default=None, repr=False, eq=False)

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
        return PrintStatement(args, sep, end, node)


Statement = PassStatement | ReturnStatement | PrintStatement


def parse_statement(node: PTNode, scope: Scope) -> Statement:
    match node.type:
        case "pass_statement":
            return PassStatement(node)
        case "return_statement":
            return ReturnStatement.make(node, scope)
        case "print_statement":
            return PrintStatement.make(node, scope)
        case unexpected:
            raise UnexpectedValue(unexpected, node.pos)


@attrs.define
class Reference:
    parts: list[str]
    scope: Scope | None = attrs.field(default=None, repr=False, eq=False)
    node: PTNode | None = attrs.field(default=None, repr=False, eq=False)

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> Reference:
        parts = [s.text for s in node.named_children]
        return Reference(parts, scope, node)

    def resolve(self) -> Any:
        assert self.scope is not None
        assert self.node is not None

        head, tail = self.parts[0], self.parts[1:]
        _, v = self.scope.resolve(head, self.node)
        for part in tail:
            _, v = v.scope.resolve(part, self.node)

        return v


Expression = int | float | bool | Reference


def parse_expression(node: PTNode, scope: Scope) -> Expression:
    match node.type:
        case "integer" | "float":
            return make_number(node)
        case "boolean":
            return make_bool(node)
        case "reference":
            return Reference.make(node, scope)
        case unexpected:
            raise UnexpectedValue(unexpected, node.pos)


def expression_type(e: Expression) -> str:
    match e:
        case bool():
            return "bool"
        case int():
            return "i64"
        case float():
            return "f64"
        case Reference():
            r = e.resolve()
            match r:
                case Config():
                    return r.type.name
                case unexpcted:
                    return f"unexpcted type {unexpcted}"
        case unexpcted:
            return f"unexpcted type {unexpcted}"


@attrs.define
class Param:
    name: str
    type: str

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> Param:
        name = node.field("name").text
        type_node = node.field("type")
        match scope.resolve(type_node.text, type_node):
            case [_, (BuiltinType() | EnumType())]:
                type = type_node.text
            case _:
                raise Error(
                    "Invalid type",
                    f"{type_node.text} is not a built in or enumerated type",
                    type_node.pos,
                )

        obj = cls(name, type)
        scope.define(name, obj, True, node)
        return obj


@attrs.define
class Function:
    name: str
    params: list[Param]
    return_: str | None
    body: list[Statement]
    scope: Scope | None = attrs.field(default=None, repr=False, eq=False)
    node: PTNode | None = attrs.field(default=None, repr=False, eq=False)

    @classmethod
    def make(cls, node: PTNode, parent_scope: Scope) -> Function:
        name = node.field("name").text
        scope = Scope(name=name, parent=parent_scope)

        params = []
        for child in node.field("params").named_children:
            params.append(Param.make(child, scope))

        return_node = node.maybe_field("return_")
        if return_node is None:
            return_ = None
        else:
            match scope.resolve(return_node.text, return_node):
                case [_, (BuiltinType(), EnumType())]:
                    return_ = return_node.text
                case _:
                    raise Error(
                        "Invalid type",
                        f"{return_node.text} is not a built in or enumerated type",
                        return_node.pos,
                    )

        body = []
        for child in node.field("body").named_children:
            if child.type in ("pass_statement", "return_statement", "print_statement"):
                body.append(parse_statement(child, scope))

        obj = Function(name, params, return_, body, scope, node)
        parent_scope.define(name, obj, True, node)
        return obj


@attrs.define
class Source:
    module: str
    configs: list[Config]
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
        add_builtins(scope, node)

        configs: list[Config] = []
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
                case "enum":
                    enums.append(EnumType.make(child, scope))
                case "node":
                    if node_table is None:
                        node_table = NodeTable.make(child, scope)
                    else:
                        raise Error(
                            "Multiple node table",
                            "Node table is multiply defined",
                            child.pos,
                        )
                case "edge":
                    if edge_table is None:
                        edge_table = EdgeTable.make(child, scope)
                    else:
                        raise Error(
                            "Multiple edge table",
                            "Edge table is multiply defined",
                            child.pos,
                        )
                case "distributions":
                    for d in child.named_children:
                        distributions.append(parse_distribution(d, scope))
                case "contagion":
                    contagions.append(Contagion.make(child, scope))
                case "function":
                    func = Function.make(child, scope)
                    if func.name == "main":
                        if main_function is None:
                            main_function = func
                        else:
                            raise Error(
                                "Multiple main",
                                "Main function is multiply defined",
                                child.pos,
                            )
                    else:
                        functions.append(func)

        if node_table is None:
            raise Error("Node undefined", "Node table was not defined", node.pos)
        if edge_table is None:
            raise Error("Edge undefined", "Edge table was not defined", node.pos)

        if main_function is None:
            raise Error("Main undefined", "No main function was defined", node.pos)
        if main_function.params:
            raise Error(
                "Bad main", "Main function must not have any parameters", node.pos
            )
        if main_function.return_ is not None:
            raise Error(
                "Bad main", "Main function must not have a return type", node.pos
            )

        for c in contagions:
            node_table.add_contagion_fields(c)

        return cls(
            module=module,
            configs=configs,
            enums=enums,
            node_table=node_table,
            edge_table=edge_table,
            distributions=distributions,
            contagions=contagions,
            functions=functions,
            main_function=main_function,
            scope=scope,
        )


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
