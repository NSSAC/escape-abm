"""Abstract syntax tree."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable, Iterable, overload
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
from .misc import SourcePosition, CodeError, CodeErrorList, SemanticError
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


def assert_exactly_one(
    parent: PTNode, children: Iterable[PTNode | AstNode], error_desc: str
):
    children = list(children)

    if len(children) == 0:
        raise SemanticError(error_desc, parent.pos)

    elif len(children) == 1:
        return

    else:
        errors: list[SemanticError] = []
        for child in children:
            errors.append(SemanticError(error_desc, child.pos))
        raise CodeErrorList(error_desc, errors)


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
    except KeyError as e:
        raise RuntimeError(f"Parser for node type '{node.type}' is not defined") from e

    try:
        ret = node_parser(node)
    except CodeError as e:
        if e.pos is None:
            e.pos = node.pos
        raise e
    except CodeErrorList:
        raise
    except Exception as e:
        raise SemanticError(f"Failed to parse '{node.type}'", node.pos) from e

    if not isinstance(ret, type):
        raise TypeError(f"Expected object of type {type}; got {type(ret)}", node.pos)

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
        names = [s.text for s in node.named_children()]
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
    def parse(node: PTNode) -> FunctionCall:
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

    @staticmethod
    def parse_parameter(node: PTNode) -> Variable:
        name = node.field("name").text
        type = get_type(node.field("type").text)
        obj = Variable(name=name, type=type, pos=node.pos)
        scope().define(name, obj)
        return obj

    @staticmethod
    def parse_lambda_parameter(node: PTNode, expected_type: Type) -> Variable:
        name = node.field("name").text
        type = node.maybe_field("type")
        if type is None:
            type = expected_type
        else:
            type = get_type(node.field("type").text)
        obj = Variable(name=name, type=type, pos=node.pos)
        scope().define(name, obj)
        return obj

    @staticmethod
    def parse_assignment(name: str, node: PTNode) -> Variable:
        type = node.maybe_field("type")
        if type is None:
            raise SemanticError(
                f"Type of local variable '{name}' not specified", node.pos
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
        call = parse(node.named_children()[0], FunctionCall)
        return CallStatement(call=call, pos=node.pos)


@dataclass
class ReturnStatement(AstNode):
    expression: Expression

    @parser("return_statement")
    @staticmethod
    def parse(node: PTNode) -> ReturnStatement:
        expression = parse_expression(node.named_children()[0])
        return ReturnStatement(expression=expression, pos=node.pos)

    @staticmethod
    def from_expression(node: PTNode) -> ReturnStatement:
        expression = parse_expression(node)
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
class FilterClause(AstNode):
    function: Expression | LambdaFunction

    @parser("filter_clause")
    @staticmethod
    def parse(node: PTNode) -> FilterClause:
        function = parse_expression_or_lambda(
            node.field("function"),
        )
        obj = FilterClause(function=function, pos=node.pos)
        return obj


@dataclass
class SampleClause(AstNode):
    is_absolute: bool
    amount: Expression

    @parser("sample_clause")
    @staticmethod
    def parse(node: PTNode) -> SampleClause:
        is_absolute = node.field("type").text == "ABSOLUTE"
        amount = parse_expression(node.field("amount"))
        obj = SampleClause(is_absolute=is_absolute, amount=amount, pos=node.pos)
        return obj


@dataclass
class ApplyClause(AstNode):
    function: Expression | LambdaFunction

    @parser("apply_clause")
    @staticmethod
    def parse(node: PTNode) -> ApplyClause:
        function = parse_expression_or_lambda(
            node.field("function"),
        )
        obj = ApplyClause(function=function, pos=node.pos)
        return obj


@dataclass
class ReduceClause(AstNode):
    lvalue: Reference
    operator: str
    function: Expression | LambdaFunction

    @parser("reduce_clause")
    @staticmethod
    def parse(node: PTNode) -> ReduceClause:
        lvalue = parse(node.field("lvalue"), Reference)
        operator = node.field("operator").text
        function = parse_expression_or_lambda(
            node.field("function"),
        )

        obj = ReduceClause(
            lvalue=lvalue, operator=operator, function=function, pos=node.pos
        )
        return obj


@dataclass
class ParallelStatement(AstNode):
    table: str
    filter_clause: FilterClause | None
    sample_clause: SampleClause | None
    apply_clause: ApplyClause | None
    reduce_clauses: list[ReduceClause]

    @parser("parallel_statement")
    @staticmethod
    def parse(node: PTNode) -> ParallelStatement:
        table = node.field("table").text

        expected_lambda_type = FunctionType(
            params=(get_type(table),), return_=get_type("bool")
        )
        scope().define("_expected_lambda_type", expected_lambda_type)
        filter_clause_list: list[FilterClause] = []
        for clause in node.named_children("filter_clause"):
            filter_clause_list.append(parse(clause, FilterClause))
        assert len(filter_clause_list) <= 1, "Got multiple filter clauses"
        if len(filter_clause_list) == 1:
            filter_clause = filter_clause_list[0]
        else:
            filter_clause = None
        scope().undef("_expected_lambda_type")

        sample_clause_list: list[SampleClause] = []
        for clause in node.named_children("sample_clause"):
            sample_clause_list.append(parse(clause, SampleClause))
        assert len(sample_clause_list) <= 1, "Got multiple sample clauses"
        if len(sample_clause_list) == 1:
            sample_clause = sample_clause_list[0]
        else:
            sample_clause = None

        expected_lambda_type = FunctionType(
            params=(get_type(table),), return_=get_type("void")
        )
        scope().define("_expected_lambda_type", expected_lambda_type)
        apply_clause_list: list[ApplyClause] = []
        for clause in node.named_children("apply_clause"):
            apply_clause_list.append(parse(clause, ApplyClause))
        assert len(apply_clause_list) <= 1, "Got multiple apply clauses"
        if len(apply_clause_list) == 1:
            apply_clause = apply_clause_list[0]
        else:
            apply_clause = None
        scope().undef("_expected_lambda_type")

        expected_lambda_type = FunctionType(
            params=(get_type(table),), return_=get_type("float")
        )
        scope().define("_expected_lambda_type", expected_lambda_type)
        reduce_clauses: list[ReduceClause] = []
        for clause in node.named_children("reduce_clause"):
            reduce_clauses.append(parse(clause, ReduceClause))
        scope().undef("_expected_lambda_type")

        return ParallelStatement(
            table=table,
            filter_clause=filter_clause,
            sample_clause=sample_clause,
            apply_clause=apply_clause,
            reduce_clauses=reduce_clauses,
            pos=node.pos,
        )


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
                params.append(Variable.parse_parameter(child))

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
class LambdaFunction(AstNode):
    name: str
    expected_type: FunctionType
    type: FunctionType
    params: list[Variable]
    local_variables: list[Variable]
    body: list[Statement]
    scope: Scope = field(repr=False)

    @parser("lambda_function")
    @staticmethod
    def parse(node: PTNode) -> LambdaFunction:
        name = f"lambda_function_{node.pos.line}_{node.pos.col}"
        params: list[Variable] = []
        local_variables: list[Variable] = []

        expected_type = scope().resolve("_expected_lambda_type", FunctionType)

        return_type = node.maybe_field("type")
        if return_type is None:
            return_type = expected_type.return_
        else:
            return_type = get_type(return_type.text)

        with new_scope(name):
            fn_scope = scope()

            for child, etype in zip(node.fields("parameter"), expected_type.params):
                params.append(Variable.parse_lambda_parameter(child, etype))

            type = FunctionType(
                params=tuple(p.type for p in params), return_=return_type
            )

            fn_scope.define("_local_variables", local_variables)

            body: list[Statement] = []
            for child in node.fields("body"):
                body.append(parse_statement(child))
            return_expr = node.maybe_field("return_expression")
            if return_expr:
                body.append(ReturnStatement.from_expression(return_expr))

            fn_scope.undef("_local_variables")

        return LambdaFunction(
            name=name,
            expected_type=expected_type,
            type=type,
            params=params,
            local_variables=local_variables,
            body=body,
            scope=fn_scope,
            pos=node.pos,
        )


def parse_expression_or_lambda(node: PTNode) -> Expression | LambdaFunction:
    return parse(node, Expression | LambdaFunction)


@dataclass
class NodeField(AstNode):
    name: str
    type: Type
    is_node_key: bool
    is_static: bool
    save_to_output: bool

    @parser("node_field")
    @staticmethod
    def parse(node: PTNode) -> NodeField:
        name = node.field("name").text
        type = get_type(node.field("type").text)

        annotations = [c.text for c in node.fields("annotation")]
        is_node_key = "node key" in annotations
        is_static = "static" in annotations or is_node_key
        save_to_output = "save" in annotations

        if save_to_output and is_static:
            raise SemanticError("Static field can't be saved to output.", node.pos)

        obj = NodeField(
            name=name,
            type=type,
            is_node_key=is_node_key,
            is_static=is_static,
            save_to_output=save_to_output,
            pos=node.pos,
        )

        scope().define(name, obj)
        return obj


@dataclass
class NodeTable(AstNode):
    fields: list[NodeField]
    key: NodeField
    scope: Scope

    @parser("node")
    @staticmethod
    def parse(node: PTNode) -> NodeTable:
        with new_scope("node_table"):
            fields = [parse(child, NodeField) for child in node.named_children()]
            table_scope = scope()

        key_list = [f for f in fields if f.is_node_key]
        assert_exactly_one(
            node, key_list, "One and only one node key field must be specified."
        )
        key = key_list[0]

        obj = NodeTable(fields=fields, key=key, scope=table_scope, pos=node.pos)
        return obj


@dataclass
class EdgeField(AstNode):
    name: str
    type: Type
    is_target_node_key: bool
    is_source_node_key: bool
    is_static: bool
    save_to_output: bool

    @parser("edge_field")
    @staticmethod
    def parse(node: PTNode) -> EdgeField:
        name = node.field("name").text
        type = get_type(node.field("type").text)

        annotations = [c.text for c in node.fields("annotation")]
        is_source_node_key = "source node key" in annotations
        is_target_node_key = "target node key" in annotations
        is_static = "static" in annotations or is_source_node_key or is_target_node_key
        save_to_output = "save" in annotations

        if is_static and save_to_output:
            raise SemanticError("Static field can't be saved to output.", node.pos)

        obj = EdgeField(
            name=name,
            type=type,
            is_target_node_key=is_target_node_key,
            is_source_node_key=is_source_node_key,
            is_static=is_static,
            save_to_output=save_to_output,
            pos=node.pos,
        )

        scope().define(name, obj)
        return obj


@dataclass
class EdgeTable(AstNode):
    fields: list[EdgeField]
    target_node_key: EdgeField
    source_node_key: EdgeField
    scope: Scope

    @parser("edge")
    @staticmethod
    def parse(node: PTNode) -> EdgeTable:
        with new_scope("edge_table"):
            fields = [parse(child, EdgeField) for child in node.named_children()]
            table_scope = scope()

        target_key_list = [f for f in fields if f.is_target_node_key]
        assert_exactly_one(
            node,
            target_key_list,
            "One and only one target node key field must be specified.",
        )
        target_key = target_key_list[0]

        source_key_list = [f for f in fields if f.is_source_node_key]
        assert_exactly_one(
            node,
            source_key_list,
            "One and only one source node key field must be specified.",
        )
        source_key = source_key_list[0]

        obj = EdgeTable(
            fields=fields,
            target_node_key=target_key,
            source_node_key=source_key,
            scope=table_scope,
            pos=node.pos,
        )
        return obj


@dataclass
class Transition(AstNode):
    entry: Reference
    exit: Reference

    @parser("transition")
    @staticmethod
    def parse(node: PTNode) -> Transition:
        entry = parse(node.field("entry"), Reference)
        exit = parse(node.field("exit"), Reference)
        return Transition(entry=entry, exit=exit, pos=node.pos)


@dataclass
class Transmission(AstNode):
    contact: Reference
    entry: Reference
    exit: Reference

    @parser("transmission")
    @staticmethod
    def parse(node: PTNode) -> Transmission:
        contact = parse(node.field("contact"), Reference)
        entry = parse(node.field("entry"), Reference)
        exit = parse(node.field("exit"), Reference)
        return Transmission(contact=contact, entry=entry, exit=exit, pos=node.pos)


@dataclass
class Contagion(AstNode):
    name: str
    type: Type
    transitions: list[Transition]
    transmissions: list[Transmission]
    transition_rate: Expression | LambdaFunction
    dwell_time: Expression | LambdaFunction
    susceptibility: Expression | LambdaFunction
    infectivity: Expression | LambdaFunction
    transmissibility: Expression | LambdaFunction
    # scope: Scope

    @parser("contagion")
    @staticmethod
    def parse(node: PTNode) -> Contagion:
        name = node.field("name").text
        type = get_type(node.field("type").text)
        transitions = [
            parse(child, Transition) for child in node.named_children("transition")
        ]
        transmissions = [
            parse(child, Transmission) for child in node.named_children("transmission")
        ]

        expected_lambda_type = FunctionType(
            params=(get_type("node"), type), return_=get_type("float")
        )
        scope().define("_expected_lambda_type", expected_lambda_type)
        transition_rate_list: list[Expression | LambdaFunction] = []
        for child in node.fields("transition_rate"):
            transition_rate_list.append(parse_expression_or_lambda(child))
        assert_exactly_one(
            node,
            transition_rate_list,
            "One and only one transition rate function must be provided.",
        )
        transition_rate = transition_rate_list[0]
        scope().undef("_expected_lambda_type")

        expected_lambda_type = FunctionType(
            params=(get_type("node"), type), return_=get_type("float")
        )
        scope().define("_expected_lambda_type", expected_lambda_type)
        dwell_time_list: list[Expression | LambdaFunction] = []
        for child in node.fields("dwell_time"):
            dwell_time_list.append(parse_expression_or_lambda(child))
        assert_exactly_one(
            node,
            dwell_time_list,
            "One and only one dwell time function must be provided.",
        )
        dwell_time = dwell_time_list[0]
        scope().undef("_expected_lambda_type")

        expected_lambda_type = FunctionType(
            params=(get_type("node"),), return_=get_type("float")
        )
        scope().define("_expected_lambda_type", expected_lambda_type)
        susceptibility_list: list[Expression | LambdaFunction] = []
        for child in node.fields("susceptibility"):
            susceptibility_list.append(parse_expression_or_lambda(child))
        assert_exactly_one(
            node,
            susceptibility_list,
            "One and only one susceptibility function must be provided.",
        )
        susceptibility = susceptibility_list[0]
        scope().undef("_expected_lambda_type")

        expected_lambda_type = FunctionType(
            params=(get_type("node"),), return_=get_type("float")
        )
        scope().define("_expected_lambda_type", expected_lambda_type)
        infectivity_list: list[Expression | LambdaFunction] = []
        for child in node.fields("infectivity"):
            infectivity_list.append(parse_expression_or_lambda(child))
        assert_exactly_one(
            node,
            infectivity_list,
            "One and only one infectivity function must be provided.",
        )
        infectivity = infectivity_list[0]
        scope().undef("_expected_lambda_type")

        expected_lambda_type = FunctionType(
            params=(get_type("edge"),), return_=get_type("float")
        )
        scope().define("_expected_lambda_type", expected_lambda_type)
        transmissibility_list: list[Expression | LambdaFunction] = []
        for child in node.fields("transmissibility"):
            transmissibility_list.append(parse_expression_or_lambda(child))
        assert_exactly_one(
            node,
            transmissibility_list,
            "One and only one transmissibility function must be provided.",
        )
        transmissibility = transmissibility_list[0]
        scope().undef("_expected_lambda_type")

        obj = Contagion(
            name=name,
            type=type,
            transition_rate=transition_rate,
            dwell_time=dwell_time,
            transitions=transitions,
            transmissions=transmissions,
            susceptibility=susceptibility,
            infectivity=infectivity,
            transmissibility=transmissibility,
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
    node_table: NodeTable
    edge_table: EdgeTable
    contagions: list[Contagion]

    @staticmethod
    def parse(module: str, root: PTNode) -> Source:
        clear_scope()
        setup_type_system()

        with new_scope(name="source"):
            for type in visible_types():
                scope().define(type.name, type)

            enum_type_defns = [
                parse(enum, EnumTypeDefn) for enum in root.named_children("enum")
            ]

            globals = [
                parse(gvar, GlobalVariable) for gvar in root.named_children("global")
            ]

            functions = [
                parse(func, Function) for func in root.named_children("function")
            ]

            node_table_list = [
                parse(table, NodeTable) for table in root.named_children("node")
            ]
            assert_exactly_one(
                root, node_table_list, "One and only one node table must be defined"
            )
            node_table = node_table_list[0]

            edge_table_list = [
                parse(table, EdgeTable) for table in root.named_children("edge")
            ]
            assert_exactly_one(
                root, edge_table_list, "One and only one edge table must be defined"
            )
            edge_table = edge_table_list[0]

            contagions = [
                parse(contg, Contagion) for contg in root.named_children("contagion")
            ]

        return Source(
            module=module,
            enum_type_defns=enum_type_defns,
            globals=globals,
            functions=functions,
            node_table=node_table,
            edge_table=edge_table,
            contagions=contagions,
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
    except CodeErrorList:
        raise
    except Exception as e:
        raise RuntimeError("Failed create syntax tree") from e


@click.command()
@simulation_file_option
def print_ast(simulation_file: Path):
    """Print the AST."""
    file_bytes = simulation_file.read_bytes()
    pt = mk_pt(str(simulation_file), file_bytes)
    ast = mk_ast(simulation_file, pt)
    rich.print(ast)
