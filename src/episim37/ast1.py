"""Abstract syntax tree v1."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import attrs
import click

import rich
import rich.markup

from .parse_tree import mk_pt, PTNode, ParseTreeConstructionError


class ASTConstructionError(Exception):
    def __init__(self, type: str, explanation: str, node: PTNode):
        super().__init__()
        self.type = type
        self.explanation = explanation
        self.node = node.node

    def __str__(self):
        return "Failed to construct AST"

    def rich_print(self, file_path: Path, file_bytes: bytes):
        fpath = str(file_path)
        etype = self.type
        line = self.node.start_point[0] + 1
        col = self.node.start_point[1] + 1
        expl = rich.markup.escape(self.explanation)

        rich.print(f"[red]{self}[/red]")
        rich.print(f"[red]{etype}[/red]:[yellow]{fpath}[/yellow]:{line}:{col}: {expl}")
        start = self.node.start_byte
        end = self.node.end_byte
        error_part = file_bytes[start:end].decode()
        print(error_part)


class UnexpectedValue(ASTConstructionError):
    def __init__(self, unexpected, node: PTNode):
        super().__init__("Unexpected value", str(unexpected), node)


Error = ASTConstructionError


@attrs.define
class Scope:
    name: str
    names: dict[str, Any] = attrs.field(factory=dict)
    consts: set[str] = attrs.field(factory=set)
    parent: Scope | None = attrs.field(default=None)

    def __rich_repr__(self):
        yield "name", self.name
        yield "names", {k: type(v).__name__ for k, v in self.names.items()}
        if self.consts:
            yield "consts", self.consts
        if self.parent is not None:
            yield "parent", self.parent.name

    def resolve(self, k: str, node: PTNode) -> tuple[Scope, Any]:
        if k in self.names:
            return self, self.names[k]
        elif self.parent is not None:
            return self.parent.resolve(k, node)

        raise Error("Undefined reference", f"{k} is not defined", node)

    def define(self, k: str, v: Any, const: bool, node: PTNode):
        if k in self.names:
            raise Error(
                "Already defined",
                f"{k} has been already defined",
                node,
            )

        self.names[k] = v
        if const:
            self.consts.add(k)

    def update(self, k: str, v: Any, node: PTNode):
        scope, _ = self.resolve(k, node)
        if k in scope.consts:
            raise Error(
                "Constant update",
                f"{k} can't be updated",
                node,
            )
        scope.names[k] = v


def smallest_int_type(max_val: int) -> str | None:
    if max_val < 2**8:
        type = "u8"
    elif max_val < 2**16:
        type = "u16"
    elif max_val < 2**32:
        type = "u32"
    elif max_val < 2**32:
        type = "u64"
    else:
        type = None
    return type


def make_bool(text: str) -> bool:
    return text == "True"


@attrs.define
class BuiltinType:
    name: str


def add_builtin_type(type: str, scope: Scope, node: PTNode):
    scope.define(type, BuiltinType(type), True, node)


@attrs.define
class ReservedName:
    name: str


def add_reserved_name(name: str, scope: Scope, node: PTNode):
    assert scope.define(name, ReservedName(name), True, node)


def add_builtins(scope: Scope, node: PTNode):
    for type in ["int", "uint", "float", "bool", "node", "edge"]:
        add_builtin_type(type, scope, node)
    for type in ["u8", "u16", "u32", "u64"]:
        add_builtin_type(type, scope, node)
    for type in ["i8", "i16", "i32", "i64"]:
        add_builtin_type(type, scope, node)
    for type in ["f32", "f64"]:
        add_builtin_type(type, scope, node)


@attrs.define
class Config:
    name: str
    type: BuiltinType
    default: int | float | bool
    node: PTNode | None = attrs.field(default=None, repr=False, eq=False)

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> Config:
        name = node.field("name").text
        type_node = node.field("type")
        match scope.resolve(type_node.text, type_node):
            case _, BuiltinType() as t:
                type = t
            case _:
                raise Error(
                    "Invalid type",
                    f"{type_node.text} is not a builtin type",
                    type_node,
                )

        default_node = node.field("default")
        match default_node.type:
            case "integer":
                default = int(default_node.text)
            case "float":
                default = float(default_node.text)
            case "boolean":
                default = default_node.text == "True"
            case unexpected:
                raise UnexpectedValue(unexpected, default_node)

        obj = cls(name=name, type=type, default=default, node=node)
        scope.define(name, obj, True, node)
        return obj


class EnumConstant(str):
    pass


@attrs.define
class EnumType:
    name: str
    base_type: BuiltinType
    consts: list[str]
    node: PTNode | None = attrs.field(default=None, repr=False, eq=False)

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> EnumType:
        name = node.field("name").text
        consts = []
        for child in node.fields("const"):
            const = child.text
            scope.define(const, EnumConstant(const), True, child)
            consts.append(const)

        base_type = smallest_int_type(len(consts) - 1)
        if base_type is None:
            raise Error("Large enum", "Enum is too big", node)
        _, base_type = scope.resolve(base_type, node)

        obj = cls(name, base_type, consts, node)
        scope.define(name, obj, True, node)
        return obj


@attrs.define
class NodeField:
    name: str
    type: BuiltinType | EnumType
    is_node_key: bool
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
                    type_node,
                )

        annotations = [c.text for c in node.fields("annotation")]
        is_node_key = "node key" in annotations
        is_static = "static" in annotations or is_node_key

        obj = NodeField(name, type, is_node_key, is_static, node)
        scope.define(name, obj, True, node)
        return obj


@attrs.define
class NodeTable:
    fields: list[NodeField]
    key: NodeField
    scope: Scope | None = attrs.field(default=None, repr=False, eq=False)
    node: PTNode | None = attrs.field(default=None, repr=False, eq=False)

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
                node,
            )
        key = key[0]
        return NodeTable(fields, key, scope, node)


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
                    type_node,
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
        for child in node.named_children:
            fields.append(EdgeField.make(child, scope))

        target_key = [f for f in fields if f.is_target_node_key]
        if len(target_key) != 1:
            raise Error(
                "Edge target key error",
                "One and only one target node key field must be specified.",
                node,
            )
        target_key = target_key[0]

        source_key = [f for f in fields if f.is_source_node_key]
        if len(source_key) != 1:
            raise Error(
                "Edge source key error",
                "One and only one source node key field must be specified.",
                node,
            )
        source_key = source_key[0]

        return EdgeTable(fields, target_key, source_key, scope, node)


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
            raise UnexpectedValue(unexpected, node)


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
        case "integer":
            return int(node.text)
        case "float":
            return float(node.text)
        case "boolean":
            return make_bool(node.text)
        case "reference":
            return Reference.make(node, scope)
        case unexpected:
            raise UnexpectedValue(unexpected, node)


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
                    type_node,
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
                        return_node,
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
        node_table = None
        edge_table = None
        functions: list[Function] = []
        main_function = None
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
                            child,
                        )
                case "edge":
                    if edge_table is None:
                        edge_table = EdgeTable.make(child, scope)
                    else:
                        raise Error(
                            "Multiple edge table",
                            "Edge table is multiply defined",
                            child,
                        )
                case "function":
                    func = Function.make(child, scope)
                    if func.name == "main":
                        if main_function is None:
                            main_function = func
                        else:
                            raise Error(
                                "Multiple main",
                                "Main function is multiply defined",
                                child,
                            )
                    else:
                        functions.append(func)

        if node_table is None:
            raise Error("Node undefined", "Node table was not defined", node)
        if edge_table is None:
            raise Error("Edge undefined", "Edge table was not defined", node)

        if main_function is None:
            raise Error("Main undefined", "No main function was defined", node)
        if main_function.params:
            raise Error("Bad main", "Main function must not have any parameters", node)
        if main_function.return_ is not None:
            raise Error("Bad main", "Main function must not have a return type", node)

        return cls(
            module=module,
            configs=configs,
            enums=enums,
            node_table=node_table,
            edge_table=edge_table,
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
        pt = mk_pt(file_bytes)
        ast1 = mk_ast1(filename, pt)
        rich.print(ast1)
    except (ParseTreeConstructionError, ASTConstructionError) as e:
        e.rich_print(filename, file_bytes)
