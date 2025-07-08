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
from .misc import SourcePosition, CodeError
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
    value_: Any = None

    @cached_property
    def name(self) -> str:
        return ".".join(self.names)

    @parser("reference")
    @staticmethod
    def parse(node: PTNode) -> Reference:
        names = [s.text for s in node.named_children]
        obj = Reference(names=names, scope=scope(), pos=node.pos)
        return obj


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

    @staticmethod
    def parse_assignment(name: str, node: PTNode) -> Variable:
        type = node.maybe_field("type")
        if type is None:
            raise CodeError(
                "Type error", f"Type of local variable '{name}' not specified", node.pos
            )
        type = get_type(type.text)
        obj = Variable(name=name, type=type, pos=node.pos)

        scope().define(name, obj)
        scope().resolve("_local_variables", list).append(obj)
        return obj


@dataclass
class PassStatement(AstNode):
    @parser("pass_statement")
    @staticmethod
    def parse(node: PTNode) -> PassStatement:
        return PassStatement(pos=node.pos)


@dataclass
class CallStatement(AstNode):
    call: FunctionCall

    @parser("call_statement")
    @staticmethod
    def parse(node: PTNode) -> CallStatement:
        call = parse(node.named_children[0], FunctionCall)
        return CallStatement(call=call, pos=node.pos)


@dataclass
class ReturnStatement(AstNode):
    expression: Expression

    @parser("return_statement")
    @staticmethod
    def parse(node: PTNode) -> ReturnStatement:
        expression = parse_expression(node.named_children[0])
        return ReturnStatement(expression=expression, pos=node.pos)


@dataclass
class ElifSection(AstNode):
    condition: Expression
    body: list[Statement]

    @parser("elif_section")
    @staticmethod
    def parse(node: PTNode) -> ElifSection:
        condition = parse_expression(node.field("condition"))
        body = []
        for child in node.fields("body"):
            body.append(parse_statement(child))
        return ElifSection(condition=condition, body=body, pos=node.pos)


@dataclass
class ElseSection(AstNode):
    body: list[Statement]

    @parser("else_section")
    @staticmethod
    def parse(node: PTNode) -> ElseSection:
        body = []
        for child in node.fields("body"):
            body.append(parse_statement(child))
        return ElseSection(body=body, pos=node.pos)


@dataclass
class IfStatement(AstNode):
    condition: Expression
    body: list[Statement]
    elifs: list[ElifSection]
    else_: ElseSection | None

    @parser("if_statement")
    @staticmethod
    def parse(node: PTNode) -> IfStatement:
        condition = parse_expression(node.field("condition"))
        body: list[Statement] = []
        for child in node.fields("body"):
            body.append(parse_statement(child))
        elifs: list[ElifSection] = []
        for child in node.fields("elif"):
            elifs.append(parse(child, ElifSection))
        else_ = node.maybe_field("else")
        if else_ is not None:
            else_ = parse(else_, ElseSection)

        obj = IfStatement(
            condition=condition, body=body, elifs=elifs, else_=else_, pos=node.pos
        )
        return obj


@dataclass
class CaseSection(AstNode):
    match: Expression
    body: list[Statement]

    @parser("case_section")
    @staticmethod
    def parse(node: PTNode) -> CaseSection:
        match = parse_expression(node.field("match"))
        body: list[Statement] = []
        for child in node.fields("body"):
            body.append(parse_statement(child))
        return CaseSection(match=match, body=body, pos=node.pos)


@dataclass
class DefaultSection(AstNode):
    body: list[Statement]

    @parser("default_section")
    @staticmethod
    def parse(node: PTNode) -> DefaultSection:
        body: list[Statement] = []
        for child in node.fields("body"):
            body.append(parse_statement(child))
        return DefaultSection(body=body, pos=node.pos)


@dataclass
class SwitchStatement(AstNode):
    condition: Expression
    cases: list[CaseSection]
    default: DefaultSection | None

    @parser("switch_statement")
    @staticmethod
    def parse(node: PTNode) -> SwitchStatement:
        condition = parse_expression(node.field("condition"))
        cases: list[CaseSection] = []
        for case in node.fields("case"):
            cases.append(parse(case, CaseSection))
        default = node.maybe_field("default")
        if default is not None:
            default = parse(default, DefaultSection)

        obj = SwitchStatement(
            condition=condition, cases=cases, default=default, pos=node.pos
        )
        return obj


@dataclass
class WhileLoop(AstNode):
    condition: Expression
    body: list[Statement]

    @parser("while_loop")
    @staticmethod
    def parse(node: PTNode) -> WhileLoop:
        condition = parse_expression(node.field("condition"))
        body: list[Statement] = []
        for child in node.fields("body"):
            body.append(parse_statement(child))
        return WhileLoop(condition=condition, body=body, pos=node.pos)


@dataclass
class AssignmentStatement(AstNode):
    lvalue: Reference
    rvalue: Expression
    variable: Variable | None

    @parser("assignment_statement")
    @staticmethod
    def parse(node: PTNode) -> AssignmentStatement:
        lvalue = parse(node.field("lvalue"), Reference)
        rvalue = parse_expression(node.field("rvalue"))

        variable = None
        if len(lvalue.names) == 1:
            name = lvalue.names[0]
            if not lvalue.scope.is_defined(name):
                variable = Variable.parse_assignment(name, node)

        obj = AssignmentStatement(
            lvalue=lvalue, rvalue=rvalue, variable=variable, pos=node.pos
        )
        return obj


@dataclass
class UpdateStatement(AstNode):
    lvalue: Reference
    operator: str
    rvalue: Expression

    @parser("update_statement")
    @staticmethod
    def parse(node: PTNode) -> UpdateStatement:
        lvalue = parse(node.field("lvalue"), Reference)
        operator = node.field("operator").text
        rvalue = parse_expression(node.field("rvalue"))
        obj = UpdateStatement(
            lvalue=lvalue, operator=operator, rvalue=rvalue, pos=node.pos
        )
        return obj


@dataclass
class ParallelStatement(AstNode):
    table: str
    # filter_clause: FilterClause | None
    # sample_clause: SampleClause | None
    # apply_clause: ApplyClause | None
    # reduce_clauses: list[ReduceClause]

    @parser("parallel_statement")
    @staticmethod
    def parse(node: PTNode) -> ParallelStatement:
        table = node.field("table").text
        return ParallelStatement(table=table, pos=node.pos)


Statement = (
    PassStatement
    | CallStatement
    | ReturnStatement
    | IfStatement
    | SwitchStatement
    | WhileLoop
    | AssignmentStatement
    | UpdateStatement
    | ParallelStatement
)


def parse_statement(node: PTNode) -> Statement:
    return parse(node, Statement)


@dataclass
class Function(AstNode):
    name: str
    type: FunctionType
    params: list[Variable]
    local_variables: list[Variable]
    body: list[Statement]
    scope: Scope = field(repr=False)

    @parser("function")
    @staticmethod
    def parse(node: PTNode) -> Function:
        name = node.field("name").text
        params: list[Variable] = []
        local_variables: list[Variable] = []

        return_type = node.maybe_field("type")
        if return_type is None:
            return_type = get_type("void")
        else:
            return_type = get_type(return_type.text)

        with new_scope(name):
            fn_scope = scope()

            for child in node.fields("parameter"):
                params.append(parse(child, Variable))

            type = FunctionType(
                params=tuple(p.type for p in params), return_=return_type
            )

            fn_scope.define("_local_variables", local_variables)

            body: list[Statement] = []
            for child in node.fields("body"):
                body.append(parse_statement(child))

            fn_scope.undef("_local_variables")

        obj = Function(
            name=name,
            type=type,
            params=params,
            local_variables=local_variables,
            body=body,
            scope=fn_scope,
            pos=node.pos,
        )
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
    pt = mk_pt(str(simulation_file), file_bytes)
    ast = mk_ast(simulation_file, pt)
    rich.print(ast)
