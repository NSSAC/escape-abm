"""Abstract syntax tree."""

from __future__ import annotations

from pathlib import Path
from collections import defaultdict
from contextlib import contextmanager
from types import FunctionType
from typing import Any, Callable, Self
from functools import cached_property
from enum import Enum

import rich
import click
from pydantic import BaseModel, Field

from .misc import SourcePosition, CodeError, RichException, Scope
from .types import (
    BuiltinType,
    EnumType,
    FunctionType,
    Type,
    add_base_types,
)
from .parse_tree import mk_pt, PTNode
from .click_helpers import simulation_file_option


@contextmanager
def err_desc(description: str, pos: SourcePosition | None):
    try:
        yield
    except AssertionError:
        raise CodeError("Semantic error", description, pos)


class AstNode(BaseModel):
    id_: str | None = Field(default=None, repr=False)
    pos_: SourcePosition | None = Field(default=None, repr=False)


NodeParser = Callable[[PTNode, Scope], Any]

_NODE_PARSER: dict[str, NodeParser] = {}


def parse(node: PTNode, scope: Scope) -> Any:
    try:
        node_parser = _NODE_PARSER[node.type]
    except KeyError:
        raise CodeError(
            "AST Error",
            f"Parser for node type '{node.type}' is not defined",
            node.pos,
        )

    try:
        ast_node = node_parser(node, scope)
        if isinstance(ast_node, AstNode):
            if ast_node.id_ is None:
                ast_node.id_ = f"_{node.type}_{node.pos.line}_{node.pos.col}"
            if ast_node.pos_ is None:
                ast_node.pos_ = node.pos
        return ast_node
    except CodeError:
        raise
    except Exception as e:
        estr = f"{e.__class__.__name__}: {e!s}"
        raise CodeError(
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


class Reference(AstNode):
    names: list[str]
    scope: Scope = Field(repr=False)

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> Self:
        names = [s.text for s in node.named_children]
        obj = cls(names=names, scope=scope)
        return obj

    @cached_property
    def name(self) -> str:
        return ".".join(self.names)

    @cached_property
    def value(self) -> Any:
        head, tail = self.names[0], self.names[1:]
        refs = []
        try:
            v = self.scope.resolve(head)
            refs.append(v)

            for part in tail:
                if hasattr(v, "scope"):
                    v = v.scope.resolve(part)
                    refs.append(v)
                elif hasattr(v, "type"):
                    v = v.type
                    # FIXME: !!!
                    v = v.scope.resolve(part)
                    refs.append(v)
                else:
                    raise

        except Exception as e:
            raise CodeError(
                "Reference error",
                f"Failed to resolve '{self.name}'; ({e!s})",
                self.pos_,
            )

        if len(refs) == 1:
            value = refs[0]
        else:
            value = tuple(refs)

        return value


register_parser("reference", Reference.make)


class BuiltinGlobal(BaseModel):
    name: str
    type: BuiltinType

    @classmethod
    def make(cls, name: str, type: str, scope: Scope) -> Self:
        obj = cls(name=name, type=scope.resolve(type))
        scope.define(name, obj)
        return obj


class BuiltinFunction(BaseModel):
    name: str
    type: FunctionType

    @classmethod
    def make(cls, name: str, params: list[str], return_: str, scope: Scope) -> Self:
        params_t = [scope.resolve(p) for p in params]
        return_t = scope.resolve(return_)
        obj = cls(name=name, type=FunctionType(params=params_t, return_=return_t))
        scope.define(name, obj)
        return obj


class EnumConstant(BaseModel):
    name: str
    type: EnumType


class EnumTypeDefn(AstNode):
    type: EnumType

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> Self:
        name = node.field("name").text
        consts = [child.text for child in node.fields("constant")]
        type = EnumType(name=name, consts=consts)

        obj = cls(type=type)
        for child in node.fields("constant"):
            enum_const = EnumConstant(name=child.text, type=type)
            scope.define(enum_const.name, enum_const)

        scope.define(name, type)
        return obj


register_parser("enum", EnumTypeDefn.make)


class GlobalCategory(Enum):
    Global = "global"
    Config = "config"
    Statistic = "statistic"


class GlobalVariable(AstNode):
    name: str
    type: Type
    category: GlobalCategory
    default: Expression

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> Self:
        name = node.field("name").text
        category = GlobalCategory(node.field("category").text)
        type = node.maybe_field("type")
        if type is None:
            type = ExistentialType.create()
        else:
            type = parse(type, scope).value
        default = parse(node.field("default"), scope)
        obj = cls(
            name=name,
            type=type,
            category=category,
            default=default,
        )
        scope.define(name, obj)
        return obj


register_parser("global", GlobalVariable.make)


class NodeField(AstNode):
    name: str
    type: Type
    is_node_key: bool
    is_static: bool
    save_to_output: bool

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> Self:
        name = node.field("name").text
        type = parse(node.field("type"), scope).value

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
        )
        if not is_node_key:
            scope.define(name, obj)
        return obj

    @property
    def is_used(self):
        return not self.is_node_key


register_parser("node_field", NodeField.make)


class NodeTable(AstNode):
    fields: list[NodeField]
    key: NodeField
    scope: Scope | None = Field(default=None, repr=False)

    @classmethod
    def make(cls, node: PTNode, parent_scope: Scope) -> Self:
        scope = Scope(name="node_table", parent=parent_scope)

        fields = [parse(child, scope) for child in node.named_children]
        key = [f for f in fields if f.is_node_key]
        with err_desc("One and only one node key field must be specified.", node.pos):
            assert len(key) == 1
        key = key[0]

        obj = cls(fields=fields, key=key, scope=scope)
        return obj


register_parser("node", NodeTable.make)


class EdgeField(AstNode):
    name: str
    type: Type
    is_target_node_key: bool
    is_source_node_key: bool
    is_static: bool
    save_to_output: bool

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> Self:
        name = node.field("name").text
        type = parse(node.field("type"), scope).value

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
        )
        if not (is_target_node_key or is_source_node_key):
            scope.define(name, obj)
        return obj

    @property
    def is_used(self):
        return not (self.is_target_node_key or self.is_source_node_key)


register_parser("edge_field", EdgeField.make)


class TargetNodeAccessor(BaseModel):
    type: BuiltinType


class SourceNodeAccessor(BaseModel):
    type: BuiltinType


class EdgeTable(AstNode):
    fields: list[EdgeField]
    target_node_key: EdgeField
    source_node_key: EdgeField
    source_node: SourceNodeAccessor
    target_node: TargetNodeAccessor
    scope: Scope | None = Field(default=None, repr=False)

    @classmethod
    def make(cls, node: PTNode, parent_scope: Scope) -> Self:
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

        source_node = SourceNodeAccessor(type=parent_scope.resolve("node"))
        scope.define("source_node", source_node)

        target_node = TargetNodeAccessor(type=parent_scope.resolve("node"))
        scope.define("target_node", target_node)

        obj = cls(
            fields=fields,
            target_node_key=target_key,
            source_node_key=source_key,
            source_node=source_node,
            target_node=target_node,
            scope=scope,
        )
        return obj


register_parser("edge", EdgeTable.make)


class DiscreteDist(AstNode):
    name: str
    ps: list[Expression]
    vs: list[Expression]
    type: FunctionType

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> Self:
        name = node.field("name").text
        ps = []
        vs = []
        for child in node.fields("pv"):
            p = parse(child.field("p"), scope)
            ps.append(p)

            v = parse(child.field("v"), scope)
            vs.append(v)

        type = FunctionType(params=[], return_=scope.resolve("float"))
        obj = cls(name=name, ps=ps, vs=vs, type=type)
        scope.define(name, obj)
        return obj


register_parser("discrete_dist", DiscreteDist.make)


class NormalDist(AstNode):
    name: str
    mean: Expression
    std: Expression
    min: Expression
    max: Expression
    type: FunctionType

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> Self:
        name = node.field("name").text
        mean = parse(node.field("mean"), scope)
        std = parse(node.field("std"), scope)

        min = node.maybe_field("min")
        min = -1e300 if min is None else parse(min, scope)

        max = node.maybe_field("max")
        max = 1e300 if max is None else parse(max, scope)

        type = FunctionType(params=[], return_=scope.resolve("float"))
        obj = cls(name=name, mean=mean, std=std, min=min, max=max, type=type)
        scope.define(name, obj)
        return obj


register_parser("normal_dist", NormalDist.make)


class UniformDist(AstNode):
    name: str
    low: Expression
    high: Expression
    type: FunctionType

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> Self:
        name = node.field("name").text
        low = parse(node.field("low"), scope)
        high = parse(node.field("high"), scope)
        type = FunctionType(params=[], return_=scope.resolve("float"))
        obj = cls(name=name, low=low, high=high, type=type)
        scope.define(name, obj)
        return obj


register_parser("uniform_dist", UniformDist.make)


Distribution = DiscreteDist | NormalDist | UniformDist


def parse_distributions(node: PTNode, scope: Scope) -> list[Distribution]:
    return [parse(child, scope) for child in node.named_children]


register_parser("distributions", parse_distributions)


class Transition(AstNode):
    entry: Reference
    exit: Reference
    p: Expression
    dwell: Expression

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> Self:
        entry = parse(node.field("entry"), scope)
        exit = parse(node.field("exit"), scope)
        p = node.maybe_field("p")
        if p is None:
            p = 1
        else:
            p = parse(p, scope)
        dwell = parse(node.field("dwell"), scope)
        return cls(entry=entry, exit=exit, p=p, dwell=dwell)


register_parser("transition", Transition.make)


def parse_transitions(node: PTNode, scope: Scope) -> list[Transition]:
    return [parse(child, scope) for child in node.fields("body")]


register_parser("transitions", parse_transitions)


class Transmission(AstNode):
    contact: Reference
    entry: Reference
    exit: Reference

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> Transmission:
        contact = parse(node.field("contact"), scope)
        entry = parse(node.field("entry"), scope)
        exit = parse(node.field("exit"), scope)
        return cls(contact=contact, entry=entry, exit=exit)


register_parser("transmission", Transmission.make)


def parse_transmissions(node: PTNode, scope: Scope) -> list[Transmission]:
    return [parse(child, scope) for child in node.fields("body")]


register_parser("transmissions", parse_transmissions)


def parse_contagion_state_type(node: PTNode, scope: Scope) -> Reference:
    return parse(node.field("type"), scope)


register_parser("contagion_state_type", parse_contagion_state_type)


class ContagionFuncton(Enum):
    Susceptibility = "susceptibility"
    Infectivity = "infectivity"
    Transmissibility = "transmissibility"


def parse_contagion_function(
    node: PTNode, scope: Scope
) -> tuple[ContagionFuncton, Reference]:
    type = ContagionFuncton(node.field("type").text)
    function = parse(node.field("function"), scope)
    return type, function


register_parser("contagion_function", parse_contagion_function)


class StateAccessor(BaseModel):
    type: EnumType


class Contagion(AstNode):
    name: str
    transitions: list[Transition]
    transmissions: list[Transmission]
    susceptibility: Expression
    infectivity: Expression
    transmissibility: Expression
    state: StateAccessor
    scope: Scope | None = Field(default=None, repr=False)

    @classmethod
    def make(cls, node: PTNode, parent_scope: Scope) -> Self:
        name = node.field("name").text
        scope = Scope(name=name, parent=parent_scope)

        children = defaultdict(list)
        for child in node.fields("body"):
            children[child.type].append(parse(child, scope))

        state_type = children["contagion_state_type"]
        with err_desc("State type must defined once for each contagion.", node.pos):
            assert len(state_type) == 1
        state_type = state_type[0].value

        state = StateAccessor(type=state_type)
        scope.define("state", state)

        transitions = [t for ts in children["transitions"] for t in ts]
        transmissions = [t for ts in children["transmissions"] for t in ts]
        contagion_functions = children["contagion_function"]

        susceptibility = [
            f for k, f in contagion_functions if k == ContagionFuncton.Susceptibility
        ]
        with err_desc("Susceptibility must defined once for each contagion.", node.pos):
            assert len(susceptibility) == 1
        susceptibility = susceptibility[0]

        infectivity = [
            f for k, f in contagion_functions if k == ContagionFuncton.Infectivity
        ]
        with err_desc("Infectivity must defined once for each contagion.", node.pos):
            assert len(infectivity) == 1
        infectivity = infectivity[0]

        transmissibility = [
            f for k, f in contagion_functions if k == ContagionFuncton.Transmissibility
        ]
        with err_desc(
            "Transmissibility must defined once for each contagion.", node.pos
        ):
            assert len(transmissibility) == 1
        transmissibility = transmissibility[0]

        obj = cls(
            name=name,
            transitions=transitions,
            transmissions=transmissions,
            susceptibility=susceptibility,
            infectivity=infectivity,
            transmissibility=transmissibility,
            state=state,
            scope=scope,
        )
        parent_scope.define(name, obj)
        return obj


register_parser("contagion", Contagion.make)


class Variable(AstNode):
    name: str
    type: Type
    scope: Scope | None = Field(default=None, repr=False)

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> Self:
        name = node.field("name").text
        type = node.maybe_field("type")
        if type is None:
            type = ExistentialType.create()
        else:
            type = parse(type, scope).value
        obj = cls(name=name, type=type, scope=None)
        scope.define(name, obj)
        return obj

    @classmethod
    def make_from_assignment(cls, node: PTNode, scope: Scope) -> Self:
        name = node.field("lvalue").named_children[0].text
        type = node.maybe_field("type")
        if type is None:
            type = ExistentialType.create()
        else:
            type = parse(type, scope).value
        obj = cls(name=name, type=type, scope=None)
        scope.define(name, obj)

        # This method is not called via parse
        # So we need to do this here
        obj.id_ = f"_variable_{node.pos.line}_{node.pos.col}"
        obj.pos_ = node.pos

        return obj


register_parser("parameter", Variable.make)


class Function(AstNode):
    name: str
    params: list[Variable]
    type: FunctionType
    body: list[Statement]
    scope: Scope | None = Field(default=None, repr=False)

    @classmethod
    def make(cls, node: PTNode, parent_scope: Scope) -> Self:
        name = node.field("name").text
        scope = Scope(name=name, parent=parent_scope)

        params = []
        for child in node.fields("parameter"):
            params.append(parse(child, scope))

        return_type = node.maybe_field("type")
        if return_type is None:
            return_type = ExistentialType.create()
        else:
            return_type = parse(return_type, scope).value
        type = FunctionType(params=[p.type for p in params], return_=return_type)

        body = []
        for child in node.fields("body"):
            stmt = parse(child, scope)
            body.append(stmt)

        obj = cls(name=name, params=params, type=type, body=body, scope=scope)
        parent_scope.define(name, obj)
        return obj


register_parser("function", Function.make)


class LambdaFunction(AstNode):
    name: str
    params: list[Variable]
    type: FunctionType
    body: list[Statement]
    scope: Scope | None = Field(default=None, repr=False)

    @classmethod
    def make(cls, node: PTNode, parent_scope: Scope) -> Self:
        name = f"lambda_function_{node.pos.line}_{node.pos.col}"
        scope = Scope(name=name, parent=parent_scope)

        params = []
        for child in node.fields("parameter"):
            params.append(parse(child, scope))

        return_type = node.maybe_field("type")
        if return_type is None:
            return_type = ExistentialType.create()
        else:
            return_type = parse(return_type, scope).value
        type = FunctionType(params=[p.type for p in params], return_=return_type)

        body = []
        for child in node.fields("body"):
            stmt = parse(child, scope)
            body.append(stmt)

        obj = cls(name=name, params=params, type=type, body=body, scope=scope)
        parent_scope.define(name, obj)
        return obj


register_parser("lambda_function", LambdaFunction.make)


class UnaryOperator(Enum):
    Not = "not"
    Plus = "+"
    Minus = "-"


UnaryBooleanOperators = {UnaryOperator.Not}
UnaryNumericOperators = {UnaryOperator.Plus, UnaryOperator.Minus}


class UnaryExpression(AstNode):
    operator: UnaryOperator
    argument: Expression

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> Self:
        operator = UnaryOperator(node.field("operator").text)
        argument = parse(node.field("argument"), scope)
        return cls(operator=operator, argument=argument)


register_parser("unary_expression", UnaryExpression.make)


class BinaryOperator(Enum):
    Plus = "+"
    Minus = "-"
    Times = "*"
    Div = "/"

    Mod = "%"

    Or = "or"
    And = "and"

    Eq = "=="
    NotEq = "!="

    Gt = ">"
    Ge = ">="
    Le = "<="
    Lt = "<"


BinaryNumericOperators = {
    BinaryOperator.Plus,
    BinaryOperator.Minus,
    BinaryOperator.Times,
    BinaryOperator.Div,
}
BinaryIntegerOperators = {BinaryOperator.Mod}
BinaryBooleanOperators = {BinaryOperator.Or, BinaryOperator.And}
BinaryEqualityOperators = {BinaryOperator.Eq, BinaryOperator.NotEq}
BinaryOrderOperators = {
    BinaryOperator.Gt,
    BinaryOperator.Ge,
    BinaryOperator.Le,
    BinaryOperator.Lt,
}


class BinaryExpression(AstNode):
    left: Expression
    operator: BinaryOperator
    right: Expression

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> Self:
        left = parse(node.field("left"), scope)
        operator = BinaryOperator(node.field("operator").text)
        right = parse(node.field("right"), scope)
        return cls(left=left, operator=operator, right=right)


register_parser("binary_expression", BinaryExpression.make)


class ParenthesizedExpression(AstNode):
    expression: Expression

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> Self:
        expression = parse(node.field("expression"), scope)
        return cls(expression=expression)


register_parser("parenthesized_expression", ParenthesizedExpression.make)


class FunctionCall(AstNode):
    function: Reference
    args: list[Expression]

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> Self:
        function = parse(node.field("function"), scope)
        args = [parse(c, scope) for c in node.fields("arg")]
        obj = cls(
            function=function,
            args=args,
        )
        return obj


register_parser("function_call", FunctionCall.make)


class PassStatement(AstNode):
    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> Self:
        _, _ = node, scope
        return cls()


register_parser("pass_statement", PassStatement.make)


class CallStatement(AstNode):
    call: FunctionCall

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> Self:
        call = parse(node.named_children[0], scope)
        return cls(call=call)


register_parser("call_statement", CallStatement.make)


class ReturnStatement(AstNode):
    expression: Expression

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> Self:
        expression = node.named_children[0]
        expression = parse(expression, scope)
        return cls(expression=expression)


register_parser("return_statement", ReturnStatement.make)


class ElifSection(AstNode):
    condition: Expression
    body: list[Statement]

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> Self:
        condition = parse(node.field("condition"), scope)
        body = []
        for child in node.fields("body"):
            body.append(parse(child, scope))
        return cls(condition=condition, body=body)


register_parser("elif_section", ElifSection.make)


class ElseSection(AstNode):
    body: list[Statement]

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> Self:
        body = []
        for child in node.fields("body"):
            body.append(parse(child, scope))
        return cls(body=body)


register_parser("else_section", ElseSection.make)


class IfStatement(AstNode):
    condition: Expression
    body: list[Statement]
    elifs: list[ElifSection]
    else_: ElseSection | None

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> Self:
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
            condition=condition,
            body=body,
            elifs=elifs,
            else_=else_,
        )
        return obj


register_parser("if_statement", IfStatement.make)


class CaseSection(AstNode):
    match: Expression
    body: list[Statement]

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> Self:
        match = parse(node.field("match"), scope)
        body = []
        for child in node.fields("body"):
            body.append(parse(child, scope))
        return cls(match=match, body=body)


register_parser("case_section", CaseSection.make)


class DefaultSection(AstNode):
    body: list[Statement]

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> Self:
        body = []
        for child in node.fields("body"):
            body.append(parse(child, scope))
        return cls(body=body)


register_parser("default_section", DefaultSection.make)


class SwitchStatement(AstNode):
    condition: Expression
    cases: list[CaseSection]
    default: DefaultSection | None

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> Self:
        condition = parse(node.field("condition"), scope)
        cases = []
        for case in node.fields("case"):
            cases.append(parse(case, scope))
        default = node.maybe_field("default")
        if default is not None:
            default = parse(default, scope)

        obj = cls(condition=condition, cases=cases, default=default)
        return obj


register_parser("switch_statement", SwitchStatement.make)


class WhileLoop(AstNode):
    condition: Expression
    body: list[Statement]

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> Self:
        condition = parse(node.field("condition"), scope)
        body = []
        for child in node.fields("body"):
            body.append(parse(child, scope))
        return cls(condition=condition, body=body)


register_parser("while_loop", WhileLoop.make)


class AssignmentStatement(AstNode):
    lvalue: Reference
    rvalue: Expression
    variable: Variable | None

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> Self:
        lvalue = parse(node.field("lvalue"), scope)
        rvalue = parse(node.field("rvalue"), scope)

        assert isinstance(lvalue, Reference)

        variable = None
        if len(lvalue.names) == 1:
            try:
                scope.resolve(lvalue.names[0])
            except KeyError:
                variable = Variable.make_from_assignment(node, scope)

        obj = cls(lvalue=lvalue, rvalue=rvalue, variable=variable)
        return obj


register_parser("assignment_statement", AssignmentStatement.make)


class UpdateOperator(Enum):
    TimesEq = "*="
    DivEq = "/="
    ModEq = "%="
    PlusEq = "+="
    MinusEq = "+="


UpdateNumericOperators = {
    UpdateOperator.TimesEq,
    UpdateOperator.DivEq,
    UpdateOperator.PlusEq,
    UpdateOperator.MinusEq,
}
UpdateIntegeOperators = {UpdateOperator.ModEq}


class UpdateStatement(AstNode):
    lvalue: Reference
    operator: UpdateOperator
    rvalue: Expression

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> Self:
        lvalue = parse(node.field("lvalue"), scope)
        operator = UpdateOperator(node.field("operator").text)
        rvalue = parse(node.field("rvalue"), scope)
        obj = cls(lvalue=lvalue, operator=operator, rvalue=rvalue)
        return obj


register_parser("update_statement", UpdateStatement.make)


class ParallelStatementTable(Enum):
    Node = "node"
    Edge = "edge"


class ParallelStatement(AstNode):
    table: ParallelStatementTable
    filter_clause: FilterClause | None
    sample_clause: SampleClause | None
    apply_clause: ApplyClause | None
    reduce_clauses: list[ReduceClause]

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> Self:
        table = ParallelStatementTable(node.field("table").text)

        children = defaultdict(list)
        for child in node.named_children:
            children[child.type].append(parse(child, scope))

        filter_clause = children["filter_clause"]
        with err_desc(
            "Only one filter clause may be defined per parallel statement.", node.pos
        ):
            assert len(filter_clause) <= 1
        filter_clause = filter_clause[0] if filter_clause else None

        sample_clause = children["sample_clause"]
        with err_desc(
            "Only one sample clause may be defined per parallel statement.", node.pos
        ):
            assert len(sample_clause) <= 1
        sample_clause = sample_clause[0] if sample_clause else None

        apply_clause = children["apply_clause"]
        with err_desc(
            "Only one apply clause may be defined per parallel statement.", node.pos
        ):
            assert len(apply_clause) <= 1
        apply_clause = apply_clause[0] if apply_clause else None

        reduce_clauses = children["reduce_clause"]

        if reduce_clauses and apply_clause is not None:
            raise CodeError(
                "Ast Error",
                "A parallel statement may not have both apply and reduce clauses.",
                node.pos,
            )

        if not reduce_clauses and apply_clause is None:
            raise CodeError(
                "Ast Error",
                "A parallel statement must have a apply or reduce clause.",
                node.pos,
            )

        obj = cls(
            table=table,
            filter_clause=filter_clause,
            sample_clause=sample_clause,
            apply_clause=apply_clause,
            reduce_clauses=reduce_clauses,
        )
        return obj


register_parser("parallel_statement", ParallelStatement.make)


class FilterClause(AstNode):
    function: Expression

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> Self:
        function = parse(node.field("function"), scope)
        obj = cls(function=function)
        return obj


register_parser("filter_clause", FilterClause.make)


class SampleClause(AstNode):
    is_absolute: bool
    amount: Expression

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> Self:
        is_absolute = node.field("type").text == "ABSOLUTE"
        amount = parse(node.field("amount"), scope)
        obj = cls(
            is_absolute=is_absolute,
            amount=amount,
        )
        return obj


register_parser("sample_clause", SampleClause.make)


class ApplyClause(AstNode):
    function: Expression

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> Self:
        function = parse(node.field("function"), scope)
        obj = cls(function=function)
        return obj


register_parser("apply_clause", ApplyClause.make)


class ReduceOperator(Enum):
    Sum = "+"
    Product = "*"


class ReduceClause(AstNode):
    lvalue: Reference
    operator: ReduceOperator
    function: Expression

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> Self:
        lvalue = parse(node.field("lvalue"), scope)
        operator = ReduceOperator(node.field("operator").text)
        function = parse(node.field("function"), scope)
        obj = cls(lvalue=lvalue, operator=operator, function=function)
        return obj


register_parser("reduce_clause", ReduceClause.make)


Expression = (
    int
    | float
    | bool
    | str
    | Reference
    | UnaryExpression
    | BinaryExpression
    | ParenthesizedExpression
    | FunctionCall
    | LambdaFunction
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

# References which can be assigned to
# LValueRef = (
#     GlobalVariable
#     | Variable
#     | tuple[Parameter | Variable, NodeField | EdgeField]
#     | tuple[Parameter | Variable, Contagion, StateAccessor]
#     | tuple[Parameter | Variable, SourceNodeAccessor | TargetNodeAccessor, NodeField]
#     | tuple[
#         Parameter | Variable,
#         SourceNodeAccessor | TargetNodeAccessor,
#         Contagion,
#         StateAccessor,
#     ]
# )

# References which can be evaluated to get a value
# RValueRef = (
#     BuiltinGlobal
#     | EnumConstant
#     | GlobalVariable
#     | Parameter
#     | Variable
#     | NodeSet
#     | BuiltinNodeset
#     | EdgeSet
#     | BuiltinEdgeset
#     | tuple[Parameter | Variable, NodeField | EdgeField]
#     | tuple[Parameter | Variable, Contagion, StateAccessor]
#     | tuple[Parameter | Variable, SourceNodeAccessor | TargetNodeAccessor]
#     | tuple[Parameter | Variable, SourceNodeAccessor | TargetNodeAccessor, NodeField]
#     | tuple[
#         Parameter | Variable,
#         SourceNodeAccessor | TargetNodeAccessor,
#         Contagion,
#         StateAccessor,
#     ]
# )

# References which are valid types
# TValueRef = BuiltinType | EnumType

# Things that can be called
# CValueRef = BuiltinFunction | Function | Distribution


def add_builtins(scope: Scope):
    add_base_types(scope)

    scope.define("node", BuiltinType(name="node"))
    scope.define("edge", BuiltinType(name="edge"))

    BuiltinGlobal.make("NUM_TICKS", "int", scope)
    BuiltinGlobal.make("CUR_TICK", "int", scope)
    BuiltinGlobal.make("NUM_NODES", "int", scope)
    BuiltinGlobal.make("NUM_EDGES", "int", scope)

    BuiltinFunction.make("abs", ["float"], "float", scope)
    BuiltinFunction.make("min", ["float", "float"], "float", scope)
    BuiltinFunction.make("max", ["float", "float"], "float", scope)
    BuiltinFunction.make("exp", ["float"], "float", scope)
    BuiltinFunction.make("exp2", ["float"], "float", scope)
    BuiltinFunction.make("log", ["float"], "float", scope)
    BuiltinFunction.make("log2", ["float"], "float", scope)
    BuiltinFunction.make("pow", ["float", "float"], "float", scope)
    BuiltinFunction.make("sin", ["float"], "float", scope)
    BuiltinFunction.make("cos", ["float"], "float", scope)


class Source(AstNode):
    module: str
    enums: list[EnumType]
    globals: list[GlobalVariable]
    node_table: NodeTable
    edge_table: EdgeTable
    normal_dists: list[NormalDist]
    uniform_dists: list[UniformDist]
    discrete_dists: list[DiscreteDist]
    contagions: list[Contagion]
    functions: list[Function]
    intervene: Function
    scope: Scope | None = Field(default=None, repr=False)

    @classmethod
    def make(cls, module: str, node: PTNode) -> Self:
        scope = Scope(name="source")

        add_builtins(scope)

        children = defaultdict(list)
        for child in node.named_children:
            children[child.type].append(parse(child, scope))

        enums = children["enum"]
        globals = children["global"]

        node_tables = children["node"]
        with err_desc("One and only one node table must be defined", node.pos):
            assert len(node_tables) == 1
        node_table = node_tables[0]

        node_type = scope.resolve("node")
        assert isinstance(node_type, BuiltinType)
        node_type.scope = node_table.scope

        edge_tables = children["edge"]
        with err_desc("One and only one edge table must be defined", node.pos):
            assert len(edge_tables) == 1
        edge_table = edge_tables[0]

        edge_type = scope.resolve("edge")
        assert isinstance(edge_type, BuiltinType)
        edge_type.scope = edge_table.scope

        distributions = [d for ds in children["distributions"] for d in ds]
        normal_dists = [d for d in distributions if isinstance(d, NormalDist)]
        uniform_dists = [d for d in distributions if isinstance(d, UniformDist)]
        discrete_dists = [d for d in distributions if isinstance(d, DiscreteDist)]

        contagions = children["contagion"]

        functions = children["function"]

        intervenes = [f for f in functions if f.name == "intervene"]
        with err_desc("One and only one intervene function must be defined", node.pos):
            assert len(intervenes) == 1
        intervene = intervenes[0]

        # function_calls: list[FunctionCall] = get_instances(FunctionCall)
        # select_statements: list[SelectStatement] = get_instances(SelectStatement)
        # sample_statements: list[SampleStatement] = get_instances(SampleStatement)
        # apply_statements: list[ApplyStatement] = get_instances(ApplyStatement)
        # reduce_statements: list[ReduceStatement] = get_instances(ReduceStatement)

        # add_builtins(scope)

        # node_table.link_contagions(contagions)
        # edge_table.link_contagions(contagions)
        # edge_table.link_node_table(node_table)
        # Parameter.link_tables(node_table, edge_table)
        # Variable.link_tables(node_table, edge_table)

        # template_variables: list[TemplateVariable] = get_instances(TemplateVariable)

        # rich.print(scope.rich_tree())

        # At this point all references should be resolve without errors
        # ref: Reference
        # for ref in get_instances(Reference):
        #     _ = ref.value

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
            functions=functions,
            intervene=intervene,
            scope=scope,
        )


def mk_ast(filename: Path, node: PTNode) -> Source:
    """Make AST from parse tree."""
    module = filename.stem.replace(".", "_")
    try:
        return Source.make(module, node)
    except CodeError:
        raise
    except Exception as e:
        estr = f"{e.__class__.__name__}: {e!s}"
        raise CodeError(
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
