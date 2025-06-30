"""Abstract syntax tree."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable, overload
from pathlib import Path
from dataclasses import dataclass, field
from functools import cached_property

import rich
import click

from .types import (
    Type,
    EnumType,
    FunctionType,
    add_user_defined_type,
    setup_type_system,
    visible_types,
    get_type,
)
from .scope import Scope
from .misc import SourcePosition, CodeError, RichException
from .parse_tree import mk_pt, PTNode
from .click_helpers import simulation_file_option

_SCOPE: Scope | None = None


def scope() -> Scope:
    if _SCOPE is None:
        raise RuntimeError("Scope not set.")

    return _SCOPE


@contextmanager
def new_scope(name: str):
    global _SCOPE

    parent_scope = _SCOPE
    _SCOPE = Scope(name=name, parent=parent_scope)
    if parent_scope is not None:
        parent_scope.children.append(_SCOPE)

    try:
        yield
    finally:
        _SCOPE = parent_scope


def clear_scope():
    global _SCOPE

    if _SCOPE is not None:
        _SCOPE.clear()
        _SCOPE = None


@dataclass
class AstNode:
    pos: SourcePosition = field(repr=False, compare=False)


NodeParser = Callable[[PTNode], AstNode]
NodeParserDecorator = Callable[[NodeParser], NodeParser]

_NODE_PARSER: dict[str, NodeParser] = {}


def parser(type: str) -> NodeParserDecorator:
    if type in _NODE_PARSER:
        raise ValueError(f"Parser for '{type}' is already defined")

    def decorator(parser: NodeParser) -> NodeParser:
        _NODE_PARSER[type] = parser
        return parser

    return decorator


@overload
def parse[T](node: PTNode, type: type[T]) -> T: ...


@overload
def parse[T](node: PTNode, type: tuple[type[T], ...]) -> T: ...


@overload
def parse(node: PTNode, type: Any) -> Any: ...


def parse(node, type) -> Any:
    try:
        node_parser = _NODE_PARSER[node.type]
    except KeyError:
        raise CodeError(
            "AST Error",
            f"Parser for node type '{node.type}' is not defined",
            node.pos,
        )

    try:
        ret = node_parser(node)
    except CodeError as e:
        if e.pos is None:
            e.pos = node.pos
        raise e
    except Exception as e:
        raise CodeError("AST Error", f"Failed to parse '{node.type}'", node.pos) from e

    if not isinstance(ret, type):
        raise CodeError(
            "Type Error", f"Expected object of type {type}; got {type(ret)}", node.pos
        )

    return ret


@dataclass
class Literal(AstNode):
    value: int | float | bool | str

    @parser("integer")
    @staticmethod
    def parse_integer(node: PTNode) -> Literal:
        return Literal(value=int(node.text), pos=node.pos)

    @parser("float")
    @staticmethod
    def parse_float(node: PTNode) -> Literal:
        return Literal(value=float(node.text), pos=node.pos)

    @parser("boolean")
    @staticmethod
    def parse_bool(node: PTNode) -> Literal:
        return Literal(value=(node.text == "True"), pos=node.pos)

    @parser("string")
    @staticmethod
    def parse_string(node: PTNode) -> Literal:
        return Literal(value=node.text, pos=node.pos)


@dataclass
class Reference(AstNode):
    names: list[str]
    scope: Scope = field(repr=False)

    @cached_property
    def name(self) -> str:
        return ".".join(self.names)

    @parser("reference")
    @staticmethod
    def parse(node: PTNode) -> Reference:
        names = [s.text for s in node.named_children]
        obj = Reference(names=names, scope=scope(), pos=node.pos)
        return obj

    @cached_property
    def value(self) -> Any:
        head, tail = self.names[0], self.names[1:]
        refs = []
        try:
            v = self.scope.resolve(head)
            refs.append(v)

            for part in tail:
                assert hasattr(v, "type"), "Object doesn't have a type"
                assert hasattr(v.type, "fields"), "Object's type doesn't have fields"
                assert part in v.type.fields, f"Unknown field {part}"
                v = v.type.fields[part]
                refs.append(v)
        except Exception as e:
            raise CodeError(
                "Reference error",
                f"Failed to resolve '{self.name}'; ({e!s})",
                self.pos,
            )

        if len(refs) == 1:
            value = refs[0]
        else:
            value = tuple(refs)

        return value


@dataclass
class UnaryExpression(AstNode):
    operator: str
    argument: Expression

    @parser("unary_expression")
    @staticmethod
    def parse(node: PTNode) -> UnaryExpression:
        operator = node.field("operator").text
        argument = parse_expression(node.field("argument"))
        return UnaryExpression(operator=operator, argument=argument, pos=node.pos)


@dataclass
class BinaryExpression(AstNode):
    left: Expression
    operator: str
    right: Expression

    @parser("binary_expression")
    @staticmethod
    def parse(node: PTNode) -> BinaryExpression:
        left = parse_expression(node.field("left"))
        operator = node.field("operator").text
        right = parse_expression(node.field("right"))
        return BinaryExpression(left=left, operator=operator, right=right, pos=node.pos)


@dataclass
class ParenthesizedExpression(AstNode):
    expression: Expression

    @parser("parenthesized_expression")
    @staticmethod
    def parse(node: PTNode) -> ParenthesizedExpression:
        expression = parse_expression(node.field("expression"))
        return ParenthesizedExpression(expression=expression, pos=node.pos)


@dataclass
class FunctionCall(AstNode):
    function: Reference
    args: list[Expression]

    @parser("function_call")
    @staticmethod
    def make(node: PTNode) -> FunctionCall:
        function = parse(node.field("function"), Reference)
        args = [parse_expression(c) for c in node.fields("argument")]
        obj = FunctionCall(function=function, args=args, pos=node.pos)
        return obj


Expression = (
    Literal
    | Reference
    | UnaryExpression
    | BinaryExpression
    | ParenthesizedExpression
    | FunctionCall
)


def parse_expression(node: PTNode) -> Expression:
    return parse(node, Expression)


@dataclass
class EnumConstant(AstNode):
    name: str
    type: Type


@dataclass
class EnumTypeDefn(AstNode):
    type: EnumType

    @parser("enum")
    @staticmethod
    def parse(node: PTNode) -> EnumTypeDefn:
        name = node.field("name").text
        consts = tuple(child.text for child in node.fields("constant"))
        type = EnumType(name=name, consts=consts)
        add_user_defined_type(type)

        obj = EnumTypeDefn(type=type, pos=node.pos)
        for child in node.fields("constant"):
            enum_const = EnumConstant(name=child.text, type=type, pos=child.pos)
            scope().define(enum_const.name, enum_const)

        scope().define(name, type)
        return obj


@dataclass
class GlobalVariable(AstNode):
    name: str
    type: Type
    category: str
    default: Expression

    @parser("global")
    @staticmethod
    def parse(node: PTNode) -> GlobalVariable:
        name = node.field("name").text
        category = node.field("category").text
        type = get_type(node.field("type").text)
        default = parse_expression(node.field("default"))
        obj = GlobalVariable(
            name=name, type=type, category=category, default=default, pos=node.pos
        )
        scope().define(name, obj)
        return obj


@dataclass
class Variable(AstNode):
    name: str
    type: Type

    @parser("parameter")
    @staticmethod
    def parse_parameter(node: PTNode) -> Variable:
        name = node.field("name").text
        type = get_type(node.field("type").text)
        obj = Variable(name=name, type=type, pos=node.pos)
        scope().define(name, obj)
        return obj


@dataclass
class Function(AstNode):
    name: str
    type: FunctionType

    @parser("function")
    @staticmethod
    def parse(node: PTNode) -> Function:
        name = node.field("name").text
        params: list[Variable] = []

        with new_scope(name):
            for child in node.fields("parameter"):
                params.append(parse(child, Variable))

            return_type = node.maybe_field("type")
            if return_type is None:
                return_type = get_type("void")
            else:
                return_type = get_type(return_type.text)

            type = FunctionType(
                params=tuple(p.type for p in params), return_=return_type
            )

            #
            # body = []
            # for child in node.fields("body"):
            #     stmt = parse(child, scope)
            #     body.append(stmt)

        obj = Function(name=name, type=type, pos=node.pos)
        scope().define(name, obj)
        return obj


@dataclass
class Source:
    module: str
    enum_type_defns: list[EnumTypeDefn]
    globals: list[GlobalVariable]
    functions: list[Function]

    @staticmethod
    def parse(module: str, root: PTNode) -> Source:
        clear_scope()
        setup_type_system()

        enum_type_defns = []
        globals = []
        functions = []

        with new_scope(name="source"):
            for type in visible_types():
                scope().define(type.name, type)

            for enum in root.simple_query("enum"):
                enum_type_defns.append(parse(enum, EnumTypeDefn))

            for gvar in root.simple_query("global"):
                globals.append(parse(gvar, GlobalVariable))

            for func in root.simple_query("function"):
                functions.append(parse(func, Function))

        return Source(
            module=module,
            enum_type_defns=enum_type_defns,
            globals=globals,
            functions=functions,
        )


def mk_ast(filename: Path, node: PTNode) -> Source:
    """Make AST from parse tree."""
    module = filename.stem.replace(".", "_")
    try:
        return Source.parse(module, node)
    except CodeError as e:
        if e.pos is None:
            e.pos = node.pos
        raise e
    except Exception as e:
        raise CodeError("AST Error", f"Failed to parse '{node.type}'", node.pos) from e


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
