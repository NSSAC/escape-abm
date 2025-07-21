"""Abstract syntax tree."""

from __future__ import annotations

from typing import Any, Callable, Iterable, overload
from pathlib import Path
from dataclasses import dataclass, field
from functools import cached_property

import rich
import click

from .types import (
    Type,
    FunctionType,
    add_enum_type,
    add_contagion_type,
    make_fn_type,
    setup_type_system,
    get_type,
)
from .scope import Scope, get_scope, new_scope, clear_scope
from .builtin import (
    define_builtin_functions,
    define_builtin_variables,
)
from .misc import (
    SourcePosition,
    CodeError,
    CodeErrorList,
    SemanticError,
    ReferenceError,
)
from .parse_tree import mk_pt, PTNode
from .click_helpers import simulation_file_option


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
    type: Type

    @parser("integer")
    @staticmethod
    def parse_integer(node: PTNode) -> Literal:
        return Literal(value=int(node.text), type=get_type("int"), pos=node.pos)

    @parser("float")
    @staticmethod
    def parse_float(node: PTNode) -> Literal:
        return Literal(value=float(node.text), type=get_type("float"), pos=node.pos)

    @parser("boolean")
    @staticmethod
    def parse_bool(node: PTNode) -> Literal:
        return Literal(value=(node.text == "True"), type=get_type("bool"), pos=node.pos)

    @parser("string")
    @staticmethod
    def parse_string(node: PTNode) -> Literal:
        return Literal(value=node.text, type=get_type("str"), pos=node.pos)

    def visit(self) -> Iterable[AstNode]:
        yield self


@dataclass
class Reference(AstNode):
    names: list[str]
    scope: Scope = field(repr=False)
    type: Type
    ref: Any = None

    def __rich_repr__(self):
        yield "name", self.name
        yield "type", self.type.name

    @cached_property
    def name(self) -> str:
        return ".".join(self.names)

    @parser("reference")
    @staticmethod
    def parse(node: PTNode) -> Reference:
        names = [s.text for s in node.named_children()]
        obj = Reference(
            names=names, scope=get_scope(), type=get_type("_type"), pos=node.pos
        )
        return obj

    def deref(self) -> Any | tuple:
        if self.ref is not None:
            return self.ref

        try:
            refs = []
            for name in self.names:
                if not refs:
                    refs.append(self.scope.resolve(name))
                else:
                    type: Type = refs[-1].type
                    refs.append(type.get(name))
            if len(refs) == 1:
                self.ref = refs[0]
            else:
                self.ref = tuple(refs)

            return self.ref
        except Exception as e:
            raise ReferenceError(f"Failed to resolve {self.name}", self.pos) from e

    def visit(self) -> Iterable[AstNode]:
        yield self


@dataclass
class UnaryExpression(AstNode):
    operator: str
    argument: Expression
    type: Type

    @parser("unary_expression")
    @staticmethod
    def parse(node: PTNode) -> UnaryExpression:
        operator = node.field("operator").text
        argument = parse_expression(node.field("argument"))
        return UnaryExpression(
            operator=operator, argument=argument, type=get_type("_type"), pos=node.pos
        )

    def visit(self) -> Iterable[AstNode]:
        yield self
        yield from self.argument.visit()


@dataclass
class BinaryExpression(AstNode):
    left: Expression
    operator: str
    right: Expression
    type: Type

    @parser("binary_expression")
    @staticmethod
    def parse(node: PTNode) -> BinaryExpression:
        left = parse_expression(node.field("left"))
        operator = node.field("operator").text
        right = parse_expression(node.field("right"))
        return BinaryExpression(
            left=left,
            operator=operator,
            right=right,
            type=get_type("_type"),
            pos=node.pos,
        )

    def visit(self) -> Iterable[AstNode]:
        yield self
        yield from self.left.visit()
        yield from self.right.visit()


@dataclass
class ParenthesizedExpression(AstNode):
    expression: Expression
    type: Type

    @parser("parenthesized_expression")
    @staticmethod
    def parse(node: PTNode) -> ParenthesizedExpression:
        expression = parse_expression(node.field("expression"))
        return ParenthesizedExpression(
            expression=expression, type=get_type("_type"), pos=node.pos
        )

    def visit(self) -> Iterable[AstNode]:
        yield self
        yield from self.expression.visit()


@dataclass
class FunctionCall(AstNode):
    function: Reference
    args: list[Expression]
    type: Type

    @parser("function_call")
    @staticmethod
    def parse(node: PTNode) -> FunctionCall:
        function = parse(node.field("function"), Reference)
        args = [parse_expression(c) for c in node.fields("argument")]
        obj = FunctionCall(
            function=function, args=args, type=get_type("_type"), pos=node.pos
        )
        return obj

    def visit(self) -> Iterable[AstNode]:
        yield self
        yield from self.function.visit()
        for arg in self.args:
            yield from arg.visit()


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
    type: Type

    @parser("enum")
    @staticmethod
    def parse(node: PTNode) -> EnumTypeDefn:
        name = node.field("name").text
        type = get_type(name)

        obj = EnumTypeDefn(type=type, pos=node.pos)
        for child in node.fields("constant"):
            enum_const = EnumConstant(name=child.text, type=type, pos=child.pos)
            get_scope().define(enum_const.name, enum_const)

        get_scope().define(name, type)
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
        get_scope().define(name, obj)
        return obj

    def visit(self) -> Iterable[AstNode]:
        yield self
        yield from self.default.visit()


@dataclass
class Variable(AstNode):
    name: str
    type: Type

    @staticmethod
    def parse_parameter(node: PTNode) -> Variable:
        name = node.field("name").text
        type = get_type(node.field("type").text)
        obj = Variable(name=name, type=type, pos=node.pos)
        get_scope().define(name, obj)
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
        get_scope().define(name, obj)
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

        get_scope().define(name, obj)
        return obj


@dataclass
class PassStatement(AstNode):
    @parser("pass_statement")
    @staticmethod
    def parse(node: PTNode) -> PassStatement:
        return PassStatement(pos=node.pos)

    def visit(self) -> Iterable[AstNode]:
        yield self


@dataclass
class CallStatement(AstNode):
    call: FunctionCall

    @parser("call_statement")
    @staticmethod
    def parse(node: PTNode) -> CallStatement:
        call = parse(node.named_children()[0], FunctionCall)
        return CallStatement(call=call, pos=node.pos)

    def visit(self) -> Iterable[AstNode]:
        yield self
        yield from self.call.visit()


@dataclass
class ReturnStatement(AstNode):
    expression: Expression

    @parser("return_statement")
    @staticmethod
    def parse(node: PTNode) -> ReturnStatement:
        expression = parse_expression(node.named_children()[0])
        obj = ReturnStatement(expression=expression, pos=node.pos)
        return obj

    @staticmethod
    def from_expression(node: PTNode) -> ReturnStatement:
        expression = parse_expression(node)
        obj = ReturnStatement(expression=expression, pos=node.pos)
        return obj

    def visit(self) -> Iterable[AstNode]:
        yield self
        yield from self.expression.visit()


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

    def visit(self) -> Iterable[AstNode]:
        yield self
        yield from self.condition.visit()
        for stmt in self.body:
            yield from stmt.visit()


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

    def visit(self) -> Iterable[AstNode]:
        yield self
        for stmt in self.body:
            yield from stmt.visit()


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

    def visit(self) -> Iterable[AstNode]:
        yield self
        yield from self.condition.visit()
        for stmt in self.body:
            yield from stmt.visit()
        for elif_ in self.elifs:
            yield from elif_.condition.visit()
            for stmt in elif_.body:
                yield from stmt.visit()
        if self.else_ is not None:
            for stmt in self.else_.body:
                yield from stmt.visit()


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

    def visit(self) -> Iterable[AstNode]:
        yield self
        yield from self.match.visit()
        for stmt in self.body:
            yield from stmt.visit()


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

    def visit(self) -> Iterable[AstNode]:
        yield self
        for stmt in self.body:
            yield from stmt.visit()


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

    def visit(self) -> Iterable[AstNode]:
        yield self
        yield from self.condition.visit()
        for case_ in self.cases:
            yield from case_.visit()
        if self.default is not None:
            yield from self.default.visit()


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

    def visit(self) -> Iterable[AstNode]:
        yield self
        yield from self.condition.visit()
        for stmt in self.body:
            yield from stmt.visit()


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

    def visit(self) -> Iterable[AstNode]:
        yield self
        yield from self.lvalue.visit()
        yield from self.rvalue.visit()


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

    def visit(self) -> Iterable[AstNode]:
        yield self
        yield from self.lvalue.visit()
        yield from self.rvalue.visit()


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

    def visit(self) -> Iterable[AstNode]:
        yield self
        yield from self.function.visit()


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

    def visit(self) -> Iterable[AstNode]:
        yield self
        yield from self.amount.visit()


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

    def visit(self) -> Iterable[AstNode]:
        yield self
        yield from self.function.visit()


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

    def visit(self) -> Iterable[AstNode]:
        yield self
        yield from self.lvalue.visit()
        yield from self.function.visit()


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

        expected_lambda_type = make_fn_type((table,), "bool")
        get_scope().define("_expected_lambda_type", expected_lambda_type)
        filter_clause_list: list[FilterClause] = []
        for clause in node.named_children("filter_clause"):
            filter_clause_list.append(parse(clause, FilterClause))
        assert len(filter_clause_list) <= 1, "Got multiple filter clauses"
        if len(filter_clause_list) == 1:
            filter_clause = filter_clause_list[0]
        else:
            filter_clause = None
        get_scope().undef("_expected_lambda_type")

        sample_clause_list: list[SampleClause] = []
        for clause in node.named_children("sample_clause"):
            sample_clause_list.append(parse(clause, SampleClause))
        assert len(sample_clause_list) <= 1, "Got multiple sample clauses"
        if len(sample_clause_list) == 1:
            sample_clause = sample_clause_list[0]
        else:
            sample_clause = None

        expected_lambda_type = make_fn_type((table,), "void")
        get_scope().define("_expected_lambda_type", expected_lambda_type)
        apply_clause_list: list[ApplyClause] = []
        for clause in node.named_children("apply_clause"):
            apply_clause_list.append(parse(clause, ApplyClause))
        assert len(apply_clause_list) <= 1, "Got multiple apply clauses"
        if len(apply_clause_list) == 1:
            apply_clause = apply_clause_list[0]
        else:
            apply_clause = None
        get_scope().undef("_expected_lambda_type")

        expected_lambda_type = make_fn_type((table,), "float")
        get_scope().define("_expected_lambda_type", expected_lambda_type)
        reduce_clauses: list[ReduceClause] = []
        for clause in node.named_children("reduce_clause"):
            reduce_clauses.append(parse(clause, ReduceClause))
        get_scope().undef("_expected_lambda_type")

        obj = ParallelStatement(
            table=table,
            filter_clause=filter_clause,
            sample_clause=sample_clause,
            apply_clause=apply_clause,
            reduce_clauses=reduce_clauses,
            pos=node.pos,
        )

        return obj

    def visit(self) -> Iterable[AstNode]:
        yield self
        if self.filter_clause is not None:
            yield from self.filter_clause.visit()
        if self.sample_clause is not None:
            yield from self.sample_clause.visit()
        if self.apply_clause is not None:
            yield from self.apply_clause.visit()
        for clause in self.reduce_clauses:
            yield from clause.visit()


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
    type: Type
    params: list[Variable]
    local_variables: list[Variable] = field(repr=False)
    body: list[Statement]
    updates_global: bool  # transitive
    updates_table: set[str]  # transitive
    has_parallel_stmt: bool  # transitive
    scope: Scope = field(repr=False)

    @parser("function")
    @staticmethod
    def parse(node: PTNode) -> Function:
        name = node.field("name").text
        params: list[Variable] = []
        local_variables: list[Variable] = []

        return_type = node.maybe_field("type")
        if return_type is None:
            return_type = "void"
        else:
            return_type = return_type.text

        with new_scope(name):
            fn_scope = get_scope()

            for child in node.fields("parameter"):
                params.append(Variable.parse_parameter(child))

            type = make_fn_type([p.type.name for p in params], return_type)

            body: list[Statement] = []
            for child in node.fields("body"):
                body.append(parse_statement(child))

        obj = Function(
            name=name,
            type=type,
            params=params,
            local_variables=local_variables,
            body=body,
            updates_global=False,
            updates_table=set(),
            has_parallel_stmt=False,
            scope=fn_scope,
            pos=node.pos,
        )
        get_scope().define(name, obj)
        return obj

    def visit(self) -> Iterable[AstNode]:
        yield self

    def visit_body(self) -> Iterable[AstNode]:
        for stmt in self.body:
            yield from stmt.visit()


@dataclass
class LambdaFunction(AstNode):
    name: str
    type: Type
    params: list[Variable]
    body: list[Statement]
    updates_global: bool  # transitive
    updates_table: set[str]  # transitive
    has_parallel_stmt: bool  # transitive
    scope: Scope = field(repr=False)

    @parser("lambda_function")
    @staticmethod
    def parse(node: PTNode) -> LambdaFunction:
        name = f"lambda_function_{node.pos.line}_{node.pos.col}"
        params: list[Variable] = []

        expected_type = get_scope().resolve("_expected_lambda_type", FunctionType)

        return_type = node.maybe_field("type")
        if return_type is None:
            return_type = expected_type.return_.name
        else:
            return_type = return_type.text

        with new_scope(name):
            fn_scope = get_scope()

            for child, etype in zip(node.fields("parameter"), expected_type.params):
                params.append(Variable.parse_lambda_parameter(child, etype))

            type = make_fn_type([p.type.name for p in params], return_type)

            body: list[Statement] = []
            for child in node.fields("body"):
                body.append(parse_statement(child))
            return_expr = node.maybe_field("return_expression")
            if return_expr:
                return_expr = ReturnStatement.from_expression(return_expr)
                body.append(return_expr)

        return LambdaFunction(
            name=name,
            type=type,
            params=params,
            body=body,
            updates_global=False,
            updates_table=set(),
            has_parallel_stmt=False,
            scope=fn_scope,
            pos=node.pos,
        )

    def visit(self) -> Iterable[AstNode]:
        yield self

    def visit_body(self) -> Iterable[AstNode]:
        for stmt in self.body:
            yield from stmt.visit()


def parse_expression_or_lambda(node: PTNode) -> Expression | LambdaFunction:
    return parse(node, Expression | LambdaFunction)


@dataclass
class NodeField(AstNode):
    name: str
    type: Type
    is_node_key: bool
    is_static: bool
    save_to_output: bool
    is_read: bool
    is_written: bool

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
            is_read=False,
            is_written=False,
        )

        node_type = get_type("node")
        node_type.add(name, obj)
        return obj


@dataclass
class NodeTable(AstNode):
    fields: list[NodeField]
    key: NodeField

    @parser("node")
    @staticmethod
    def parse(node: PTNode) -> NodeTable:
        fields = [parse(child, NodeField) for child in node.named_children()]

        key_list = [f for f in fields if f.is_node_key]
        assert_exactly_one(
            node, key_list, "One and only one node key field must be specified."
        )
        key = key_list[0]

        obj = NodeTable(fields=fields, key=key, pos=node.pos)
        return obj


@dataclass
class EdgeField(AstNode):
    name: str
    type: Type
    is_target_node_key: bool
    is_source_node_key: bool
    is_static: bool
    save_to_output: bool
    is_read: bool
    is_written: bool

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
            is_read=False,
            is_written=False,
            pos=node.pos,
        )

        edge_type = get_type("edge")
        edge_type.add(name, obj)

        return obj


@dataclass
class NodeAccessor:
    name: str
    type: Type


@dataclass
class EdgeTable(AstNode):
    fields: list[EdgeField]
    target_node_key: EdgeField
    source_node_key: EdgeField

    @parser("edge")
    @staticmethod
    def parse(node: PTNode) -> EdgeTable:
        fields = [parse(child, EdgeField) for child in node.named_children()]

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

        edge_type = get_type("edge")
        edge_type.add("source_node", NodeAccessor("source_node", get_type("node")))
        edge_type.add("target_node", NodeAccessor("target_node", get_type("node")))

        obj = EdgeTable(
            fields=fields,
            target_node_key=target_key,
            source_node_key=source_key,
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
    state_type: Type
    state_field: NodeField
    transitions: list[Transition]
    transmissions: list[Transmission]
    transition_rate: Expression | LambdaFunction
    dwell_time: Expression | LambdaFunction
    susceptibility: Expression | LambdaFunction
    infectivity: Expression | LambdaFunction
    transmissibility: Expression | LambdaFunction

    @parser("contagion")
    @staticmethod
    def parse(node: PTNode) -> Contagion:
        name = node.field("name").text
        type = get_type(name)
        state_type = get_type(node.field("type").text)

        state_field = NodeField(
            name=f"_{name}_state",
            type=state_type,
            is_node_key=False,
            is_static=False,
            save_to_output=False,
            is_read=True,
            is_written=True,
            pos=node.pos,
        )
        type.add("state", state_field)

        transitions = [
            parse(child, Transition) for child in node.named_children("transition")
        ]
        transmissions = [
            parse(child, Transmission) for child in node.named_children("transmission")
        ]

        expected_lambda_type = make_fn_type(("node", state_type.name), "float")
        get_scope().define("_expected_lambda_type", expected_lambda_type)
        transition_rate_list: list[Expression | LambdaFunction] = []
        for child in node.fields("transition_rate"):
            transition_rate_list.append(parse_expression_or_lambda(child))
        assert_exactly_one(
            node,
            transition_rate_list,
            "One and only one transition rate function must be provided.",
        )
        transition_rate = transition_rate_list[0]
        get_scope().undef("_expected_lambda_type")

        expected_lambda_type = make_fn_type(("node", state_type.name), "float")
        get_scope().define("_expected_lambda_type", expected_lambda_type)
        dwell_time_list: list[Expression | LambdaFunction] = []
        for child in node.fields("dwell_time"):
            dwell_time_list.append(parse_expression_or_lambda(child))
        assert_exactly_one(
            node,
            dwell_time_list,
            "One and only one dwell time function must be provided.",
        )
        dwell_time = dwell_time_list[0]
        get_scope().undef("_expected_lambda_type")

        expected_lambda_type = make_fn_type(("node",), "float")
        get_scope().define("_expected_lambda_type", expected_lambda_type)
        susceptibility_list: list[Expression | LambdaFunction] = []
        for child in node.fields("susceptibility"):
            susceptibility_list.append(parse_expression_or_lambda(child))
        assert_exactly_one(
            node,
            susceptibility_list,
            "One and only one susceptibility function must be provided.",
        )
        susceptibility = susceptibility_list[0]
        get_scope().undef("_expected_lambda_type")

        expected_lambda_type = make_fn_type(("node",), return_="float")
        get_scope().define("_expected_lambda_type", expected_lambda_type)
        infectivity_list: list[Expression | LambdaFunction] = []
        for child in node.fields("infectivity"):
            infectivity_list.append(parse_expression_or_lambda(child))
        assert_exactly_one(
            node,
            infectivity_list,
            "One and only one infectivity function must be provided.",
        )
        infectivity = infectivity_list[0]
        get_scope().undef("_expected_lambda_type")

        expected_lambda_type = make_fn_type(("edge",), "float")
        get_scope().define("_expected_lambda_type", expected_lambda_type)
        transmissibility_list: list[Expression | LambdaFunction] = []
        for child in node.fields("transmissibility"):
            transmissibility_list.append(parse_expression_or_lambda(child))
        assert_exactly_one(
            node,
            transmissibility_list,
            "One and only one transmissibility function must be provided.",
        )
        transmissibility = transmissibility_list[0]
        get_scope().undef("_expected_lambda_type")

        obj = Contagion(
            name=name,
            type=type,
            state_type=state_type,
            state_field=state_field,
            transition_rate=transition_rate,
            dwell_time=dwell_time,
            transitions=transitions,
            transmissions=transmissions,
            susceptibility=susceptibility,
            infectivity=infectivity,
            transmissibility=transmissibility,
            pos=node.pos,
        )

        return obj

    def visit(self) -> Iterable[AstNode]:
        yield self
        for t in self.transitions:
            yield from t.entry.visit()
            yield from t.exit.visit()
        for t in self.transmissions:
            yield from t.contact.visit()
            yield from t.entry.visit()
            yield from t.exit.visit()
        yield from self.transition_rate.visit()
        yield from self.dwell_time.visit()
        yield from self.susceptibility.visit()
        yield from self.infectivity.visit()
        yield from self.transmissibility.visit()


@dataclass
class Source:
    module: str
    enum_type_defns: list[EnumTypeDefn]
    node_table: NodeTable
    edge_table: EdgeTable
    globals: list[GlobalVariable]
    functions: list[Function]
    contagions: list[Contagion]

    @staticmethod
    def parse(module: str, root: PTNode) -> Source:
        clear_scope()

        setup_type_system()
        for child in root.named_children("enum"):
            add_enum_type(child.field("name").text)

        for child in root.named_children("contagion"):
            add_contagion_type(child.field("name").text)

        with new_scope(name="source"):
            define_builtin_variables()
            define_builtin_functions()

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

        node_type = get_type("node")
        for contagion in contagions:
            node_table.fields.append(contagion.state_field)
            node_type.add(contagion.name, contagion)

        return Source(
            module=module,
            enum_type_defns=enum_type_defns,
            globals=globals,
            functions=functions,
            node_table=node_table,
            edge_table=edge_table,
            contagions=contagions,
        )

    def visit(self) -> Iterable[AstNode]:
        for g in self.globals:
            yield from g.visit()
        for f in self.functions:
            yield from f.visit()
        for c in self.contagions:
            yield from c.visit()


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
