"""Abstract syntax tree."""

from __future__ import annotations

from pathlib import Path
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Callable, Self
from functools import cached_property
from enum import Enum

import rich
import click
from pydantic import BaseModel, Field

from . import environ
from .scope import Scope
from .misc import SourcePosition, CodeError, RichException
from .types import BuiltinType, EnumType, DataType, FunctionType, do_setup_base_types
from .parse_tree import mk_pt, PTNode
from .click_helpers import simulation_file_option


@contextmanager
def err_desc(description: str, pos: SourcePosition | None):
    try:
        yield
    except AssertionError:
        raise CodeError("AST Error", description, pos)


def get_global_scope() -> Scope:
    scope = environ.get("global_scope")
    if isinstance(scope, Scope):
        return scope

    raise RuntimeError("global scope is not setup")


def get_current_scope() -> Scope:
    scope = environ.get("current_scope")
    if isinstance(scope, Scope):
        return scope

    raise RuntimeError("current scope is not setup")


@contextmanager
def set_current_scope(scope: Scope):
    prev_scope = environ.get("current_scope")
    environ.set("current_scope", scope)
    try:
        yield
    finally:
        environ.set("current_scope", prev_scope)


def get_builtin_type(name: str) -> BuiltinType:
    scope = get_global_scope()
    type = scope.resolve(name)

    if isinstance(type, BuiltinType):
        return type

    raise RuntimeError(f"type {name} is not defined")


class AstNode(BaseModel):
    pos: SourcePosition = Field(repr=False)


NodeParser = Callable[[PTNode], AstNode]

_NODE_PARSER: dict[str, NodeParser] = {}


def parse(node: PTNode) -> AstNode:
    try:
        node_parser = _NODE_PARSER[node.type]
    except KeyError:
        raise CodeError(
            "AST Error",
            f"Parser for node type '{node.type}' is not defined",
            node.pos,
        )

    try:
        return node_parser(node)
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


class BuiltinGlobal(BaseModel):
    name: str
    type: BuiltinType

    @classmethod
    def make(cls, name: str, type: str) -> Self:
        obj = cls(name=name, type=get_builtin_type(type))
        get_current_scope().define(name, obj)
        return obj


class BuiltinFunction(BaseModel):
    name: str
    type: FunctionType

    @classmethod
    def make(cls, name: str, params: list[str], return_: str) -> Self:
        params_t: list[DataType] = [get_builtin_type(p) for p in params]
        return_t = get_builtin_type(return_)
        obj = cls(name=name, type=FunctionType(params=params_t, return_=return_t))
        get_current_scope().define(name, obj)
        return obj


class Literal(AstNode):
    value: int | float | bool | str


register_parser("integer", lambda node: Literal(value=int(node.text), pos=node.pos))
register_parser("float", lambda node: Literal(value=float(node.text), pos=node.pos))
register_parser(
    "boolean", lambda node: Literal(value=(node.text == "True"), pos=node.pos)
)
register_parser("string", lambda node: Literal(value=node.text, pos=node.pos))


class Reference(AstNode):
    names: list[str]
    scope: Scope = Field(repr=False)

    @classmethod
    def make(cls, node: PTNode) -> Self:
        names = [s.text for s in node.named_children]
        obj = cls(names=names, scope=get_current_scope(), pos=node.pos)
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
                v = v.type.scope.resolve(part)
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


register_parser("reference", Reference.make)


class EnumConstant(AstNode):
    name: str
    type: EnumType


class EnumTypeDefn(AstNode):
    type: EnumType

    @classmethod
    def make(cls, node: PTNode) -> Self:
        scope = get_current_scope()
        name = node.field("name").text
        consts = [child.text for child in node.fields("constant")]
        type = EnumType(name=name, consts=consts)

        obj = cls(type=type, pos=node.pos)
        for child in node.fields("constant"):
            enum_const = EnumConstant(name=child.text, type=type, pos=child.pos)
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
    type: DataType
    category: GlobalCategory
    default: Expression

    @classmethod
    def make(cls, node: PTNode) -> Self:
        scope = get_current_scope()
        name = node.field("name").text
        category = GlobalCategory(node.field("category").text)
        type = scope.resolve(node.field("type").text)
        default = parse(node.field("default"))
        obj = cls(
            name=name, type=type, category=category, default=default, pos=node.pos
        )
        scope.define(name, obj)
        return obj


register_parser("global", GlobalVariable.make)


class NodeField(AstNode):
    name: str
    type: DataType
    is_node_key: bool
    is_static: bool
    save_to_output: bool

    @classmethod
    def make(cls, node: PTNode) -> Self:
        name = node.field("name").text
        type = scope.resolve(node.field("type").text)

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

        obj = cls(fields=fields, key=key, scope=scope, pos=node.pos)
        return obj


register_parser("node", NodeTable.make)


class EdgeField(AstNode):
    name: str
    type: DataType
    is_target_node_key: bool
    is_source_node_key: bool
    is_static: bool
    save_to_output: bool

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> Self:
        name = node.field("name").text
        type = scope.resolve(node.field("type").text)

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
    type: BuiltinType


class SourceNodeAccessor(BaseModel):
    type: BuiltinType


class EdgeTable(AstNode):
    fields: list[EdgeField]
    target_node_key: EdgeField
    source_node_key: EdgeField
    scope: Scope = Field(repr=False)

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
            scope=scope,
            pos=node.pos,
        )
        return obj


register_parser("edge", EdgeTable.make)


class CategoricalDist(AstNode):
    name: str
    ps: list[Expression]
    vs: list[Expression]
    type: BuiltinType

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> Self:
        name = node.field("name").text
        ps = [parse(child, scope) for child in node.fields("probability")]
        vs = [parse(child, scope) for child in node.fields("value")]

        type = environ.get("cdist_type")
        assert isinstance(type, BuiltinType)
        obj = cls(name=name, ps=ps, vs=vs, type=type, pos=node.pos)

        scope.define(name, obj)
        return obj


register_parser("cdist", CategoricalDist.make)


class Transition(AstNode):
    entry: Reference
    exit: Reference

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> Self:
        entry = parse(node.field("entry"), scope)
        exit = parse(node.field("exit"), scope)
        return cls(entry=entry, exit=exit, pos=node.pos)


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
        return cls(contact=contact, entry=entry, exit=exit, pos=node.pos)


register_parser("transmission", Transmission.make)


def parse_transmissions(node: PTNode, scope: Scope) -> list[Transmission]:
    return [parse(child, scope) for child in node.fields("body")]


register_parser("transmissions", parse_transmissions)


def parse_contagion_state_type(node: PTNode, scope: Scope) -> EnumType:
    return scope.resolve(node.field("type").text)


register_parser("contagion_state_type", parse_contagion_state_type)


class ContagionFunctionCategory(Enum):
    TransitionProbability = "transition probability"
    DwellTime = "dwell time"
    Susceptibility = "susceptibility"
    Infectivity = "infectivity"
    Transmissibility = "transmissibility"


def parse_contagion_function(
    node: PTNode, scope: Scope
) -> tuple[ContagionFunctionCategory, Expression | LambdaFunction]:
    category = ContagionFunctionCategory(node.field("category").text)
    if category in [
        ContagionFunctionCategory.TransitionProbability,
        ContagionFunctionCategory.DwellTime,
    ]:
        expected_fn_type = FunctionType(
            params=[scope.resolve("node"), environ.get("contagion_state_type")],
            return_=scope.resolve("float"),
        )
        environ.set("contagion_function_type", expected_fn_type)
    elif category in [
        ContagionFunctionCategory.Susceptibility,
        ContagionFunctionCategory.Infectivity,
    ]:
        expected_fn_type = FunctionType(
            params=[scope.resolve("node")],
            return_=scope.resolve("float"),
        )
        environ.set("contagion_function_type", expected_fn_type)
    else:  # category == Transmissibility
        expected_fn_type = FunctionType(
            params=[scope.resolve("edge")],
            return_=scope.resolve("float"),
        )
        environ.set("contagion_function_type", expected_fn_type)

    function = parse(node.field("function"), scope)

    environ.remove("contagion_function_type")

    return category, function


register_parser("contagion_function", parse_contagion_function)


class StateAccessor(BaseModel):
    type: EnumType


def _get_contagion_fn(
    functions: list, type: ContagionFunctionCategory, pos: SourcePosition
):
    func = [f for k, f in functions if k == type]
    with err_desc(
        f"{ContagionFunctionCategory.name} must defined once for each contagion.", pos
    ):
        assert len(func) == 1
    func = func[0]
    return func


class Contagion(AstNode):
    name: str
    state_type: EnumType
    transitions: list[Transition]
    transmissions: list[Transmission]
    transition_probability: Expression | LambdaFunction
    dwell_time: Expression | LambdaFunction
    susceptibility: Expression | LambdaFunction
    infectivity: Expression | LambdaFunction
    transmissibility: Expression | LambdaFunction
    scope: Scope = Field(repr=False)

    @classmethod
    def make(cls, node: PTNode, parent_scope: Scope) -> Self:
        name = node.field("name").text
        scope = Scope(name=name, parent=parent_scope)

        children = defaultdict(list)
        for child in node.fields("body"):
            children[child.type].append(parse(child, scope))

        with err_desc("State type must defined once for each contagion.", node.pos):
            assert len(children["contagion_state_type"]) == 1
        state_type = children["contagion_state_type"][0]

        state = StateAccessor(type=state_type)
        scope.define("state", state)

        transitions = [t for ts in children["transitions"] for t in ts]
        transmissions = [t for ts in children["transmissions"] for t in ts]

        contagion_functions = children["contagion_function"]
        transition_probability = _get_contagion_fn(
            contagion_functions,
            ContagionFunctionCategory.TransitionProbability,
            node.pos,
        )
        dwell_time = _get_contagion_fn(
            contagion_functions, ContagionFunctionCategory.DwellTime, node.pos
        )
        susceptibility = _get_contagion_fn(
            contagion_functions, ContagionFunctionCategory.Susceptibility, node.pos
        )
        infectivity = _get_contagion_fn(
            contagion_functions, ContagionFunctionCategory.Infectivity, node.pos
        )
        transmissibility = _get_contagion_fn(
            contagion_functions, ContagionFunctionCategory.Transmissibility, node.pos
        )

        obj = cls(
            name=name,
            state_type=state_type,
            transitions=transitions,
            transmissions=transmissions,
            transition_probability=transition_probability,
            dwell_time=dwell_time,
            susceptibility=susceptibility,
            infectivity=infectivity,
            transmissibility=transmissibility,
            scope=scope,
            pos=node.pos,
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
            type = scope.resolve(type.text)
        obj = cls(name=name, type=type, scope=None, pos=node.pos)
        scope.define(name, obj)
        return obj

    @classmethod
    def make_from_assignment(cls, node: PTNode, scope: Scope) -> Self:
        name = node.field("lvalue").named_children[0].text
        type = node.maybe_field("type")
        type = scope.resolve(node.field("type").text)
        obj = cls(name=name, type=type, scope=None, pos=node.pos)
        scope.define(name, obj)
        return obj


register_parser("parameter", Variable.make)
register_parser("lambda_parameter", Variable.make)


class Function(AstNode):
    name: str
    params: list[Variable]
    type: FunctionType
    variables: list[Variable]
    body: list[Statement]
    is_kernel: bool
    calls: list[Reference]
    scope: Scope = Field(repr=False)

    @classmethod
    def make(cls, node: PTNode, parent_scope: Scope) -> Self:
        name = node.field("name").text
        scope = Scope(name=name, parent=parent_scope)

        params = []
        for child in node.fields("parameter"):
            params.append(parse(child, scope))

        return_type = node.maybe_field("type")
        if return_type is None:
            return_type = scope.resolve("_void")
        else:
            return_type = scope.resolve(return_type.text)
        type = FunctionType(params=[p.type for p in params], return_=return_type)

        body = []
        obj = cls(
            name=name,
            params=params,
            type=type,
            variables=[],
            body=body,
            is_kernel=True,
            calls=[],
            scope=scope,
            pos=node.pos,
        )
        scope.define("_function", obj)

        for child in node.fields("body"):
            stmt = parse(child, scope)
            body.append(stmt)

        parent_scope.define(name, obj)
        return obj


register_parser("function", Function.make)


class LambdaFunction(AstNode):
    name: str
    params: list[Variable]
    type: FunctionType
    variables: list[Variable]
    body: list[Statement]
    is_kernel: bool
    calls: list[Reference]
    scope: Scope = Field(repr=False)

    @classmethod
    def make(cls, node: PTNode, parent_scope: Scope) -> Self:
        name = f"_lambda_{node.pos.line}_{node.pos.col}"
        scope = Scope(name=name, parent=parent_scope)

        params = []
        for child in node.fields("parameter"):
            params.append(parse(child, scope))

        return_type = node.maybe_field("type")
        if return_type is None:
            return_type = ExistentialType.create()
        else:
            return_type = scope.resolve(return_type.text)
        type = FunctionType(params=[p.type for p in params], return_=return_type)

        body = []
        obj = cls(
            name=name,
            params=params,
            type=type,
            variables=[],
            body=body,
            is_kernel=True,
            calls=[],
            scope=scope,
            pos=node.pos,
        )
        scope.define("_function", obj)

        for child in node.fields("body"):
            stmt = parse(child, scope)
            body.append(stmt)

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
        return cls(operator=operator, argument=argument, pos=node.pos)


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
        return cls(left=left, operator=operator, right=right, pos=node.pos)


register_parser("binary_expression", BinaryExpression.make)


class ParenthesizedExpression(AstNode):
    expression: Expression

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> Self:
        expression = parse(node.field("expression"), scope)
        return cls(expression=expression, pos=node.pos)


register_parser("parenthesized_expression", ParenthesizedExpression.make)


class FunctionCall(AstNode):
    function: Reference
    args: list[Expression]

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> Self:
        function = parse(node.field("function"), scope)
        args = [parse(c, scope) for c in node.fields("arg")]
        obj = cls(function=function, args=args, pos=node.pos)

        try:
            caller = scope.resolve("_function")
            assert isinstance(caller, Function | LambdaFunction)
            caller.calls.append(function)
        except KeyError:
            pass

        return obj


register_parser("function_call", FunctionCall.make)


class PassStatement(AstNode):
    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> Self:
        _, _ = node, scope
        return cls(pos=node.pos)


register_parser("pass_statement", PassStatement.make)


class CallStatement(AstNode):
    call: FunctionCall

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> Self:
        call = parse(node.named_children[0], scope)
        return cls(call=call, pos=node.pos)


register_parser("call_statement", CallStatement.make)


class ReturnStatement(AstNode):
    expression: Expression
    function: Function | LambdaFunction = Field(repr=False)

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> Self:
        expression = node.named_children[0]
        expression = parse(expression, scope)

        function = scope.resolve("_function")
        return cls(expression=expression, function=function, pos=node.pos)


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
        return cls(condition=condition, body=body, pos=node.pos)


register_parser("elif_section", ElifSection.make)


class ElseSection(AstNode):
    body: list[Statement]

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> Self:
        body = []
        for child in node.fields("body"):
            body.append(parse(child, scope))
        return cls(body=body, pos=node.pos)


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
            condition=condition, body=body, elifs=elifs, else_=else_, pos=node.pos
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
        return cls(match=match, body=body, pos=node.pos)


register_parser("case_section", CaseSection.make)


class DefaultSection(AstNode):
    body: list[Statement]

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> Self:
        body = []
        for child in node.fields("body"):
            body.append(parse(child, scope))
        return cls(body=body, pos=node.pos)


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

        obj = cls(condition=condition, cases=cases, default=default, pos=node.pos)
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
        return cls(condition=condition, body=body, pos=node.pos)


register_parser("while_loop", WhileLoop.make)


class AssignmentStatement(AstNode):
    lvalue: Reference
    rvalue: Expression
    function: Function | LambdaFunction = Field(repr=False)

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> Self:
        lvalue = parse(node.field("lvalue"), scope)
        rvalue = parse(node.field("rvalue"), scope)
        function = scope.resolve("_function")

        assert isinstance(lvalue, Reference)
        assert isinstance(function, Function | LambdaFunction)

        if len(lvalue.names) == 1:
            try:
                scope.resolve(lvalue.names[0])
            except KeyError:
                try:
                    variable = Variable.make_from_assignment(node, scope)
                    function.variables.append(variable)
                except Exception as e:
                    raise CodeError(
                        "AST Error",
                        f"Failed to create variable '{lvalue.names[0]}': {e}",
                        node.pos,
                    )

        obj = cls(
            lvalue=lvalue,
            rvalue=rvalue,
            function=function,
            pos=node.pos,
        )
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
    function: Function | LambdaFunction = Field(repr=False)

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> Self:
        lvalue = parse(node.field("lvalue"), scope)
        operator = UpdateOperator(node.field("operator").text)
        rvalue = parse(node.field("rvalue"), scope)
        function = scope.resolve("_function")

        obj = cls(
            lvalue=lvalue,
            operator=operator,
            rvalue=rvalue,
            function=function,
            pos=node.pos,
        )
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
        function = scope.resolve("_function")
        assert isinstance(function, Function | LambdaFunction)
        function.is_kernel = False

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
                "AST Error",
                "A parallel statement may not have both apply and reduce clauses.",
                node.pos,
            )

        if not reduce_clauses and apply_clause is None:
            raise CodeError(
                "AST Error",
                "A parallel statement must have a apply or reduce clause.",
                node.pos,
            )

        obj = cls(
            table=table,
            filter_clause=filter_clause,
            sample_clause=sample_clause,
            apply_clause=apply_clause,
            reduce_clauses=reduce_clauses,
            pos=node.pos,
        )
        return obj


register_parser("parallel_statement", ParallelStatement.make)


class FilterClause(AstNode):
    function: Expression | LambdaFunction

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> Self:
        function = parse(node.field("function"), scope)
        obj = cls(function=function, pos=node.pos)
        return obj


register_parser("filter_clause", FilterClause.make)


class SampleClause(AstNode):
    is_absolute: bool
    amount: Expression

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> Self:
        is_absolute = node.field("type").text == "ABSOLUTE"
        amount = parse(node.field("amount"), scope)
        obj = cls(is_absolute=is_absolute, amount=amount, pos=node.pos)
        return obj


register_parser("sample_clause", SampleClause.make)


class ApplyClause(AstNode):
    function: Expression | LambdaFunction

    @classmethod
    def make(cls, node: PTNode, scope: Scope) -> Self:
        function = parse(node.field("function"), scope)
        obj = cls(function=function, pos=node.pos)
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
        obj = cls(lvalue=lvalue, operator=operator, function=function, pos=node.pos)
        return obj


register_parser("reduce_clause", ReduceClause.make)


Expression = (
    Literal
    | Reference
    | UnaryExpression
    | BinaryExpression
    | ParenthesizedExpression
    | FunctionCall
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

TypeCheckable = (
    GlobalVariable
    | Distribution
    | Contagion
    | Function
    | LambdaFunction
    | Statement
    | ElifSection
    | ElseSection
    | CaseSection
    | DefaultSection
    | FilterClause
    | SampleClause
    | ApplyClause
    | ReduceClause
)

# References which can be assigned to from device code
DeviceLValueRef = (
    Variable
    | tuple[Variable, NodeField | EdgeField]
    | tuple[Variable, Contagion, StateAccessor]
    | tuple[Variable, SourceNodeAccessor | TargetNodeAccessor, NodeField]
    | tuple[
        Variable,
        SourceNodeAccessor | TargetNodeAccessor,
        Contagion,
        StateAccessor,
    ]
)

# References which can be assigned to from host code
HostLValueRef = GlobalVariable | Variable

LValueRef = DeviceLValueRef | HostLValueRef

# References which can be evaluated from device code
DeviceRValueRef = (
    BuiltinGlobal
    | EnumConstant
    | GlobalVariable
    | Variable
    | tuple[Variable, NodeField | EdgeField]
    | tuple[Variable, Contagion, StateAccessor]
    | tuple[Variable, SourceNodeAccessor | TargetNodeAccessor]
    | tuple[Variable, SourceNodeAccessor | TargetNodeAccessor, NodeField]
    | tuple[
        Variable,
        SourceNodeAccessor | TargetNodeAccessor,
        Contagion,
        StateAccessor,
    ]
)

# References which can be evaluated from host code
HostRValueRef = BuiltinGlobal | EnumConstant | GlobalVariable | Variable

RValueRef = DeviceRValueRef | DeviceLValueRef

# References to things that can be called
CValueRef = BuiltinFunction | Function | Distribution

# All referenceable things
AllRef = LValueRef | RValueRef | CValueRef


def add_builtins(scope: Scope):
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
    enums: list[EnumTypeDefn]
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

        add_base_types(scope)
        add_builtins(scope)
        type_graph = make_type_graph()

        enums: list[EnumTypeDefn] = []
        for child in node.simple_query("enum"):
            enum = parse(child, scope)
            type_graph.add_edge(enum.type.name, "_enum")
            enums.append(enum)

        children = defaultdict(list)
        for child in node.named_children:
            if child.type != "enum":
                children[child.type].append(parse(child, scope))

        globals = children["global"]

        with err_desc("One and only one node table must be defined", node.pos):
            assert len(children["node"]) == 1
        node_table = children["node"][0]
        node_type = scope.resolve("node")
        assert isinstance(node_type, BuiltinType)
        node_type.scope = node_table.scope
        assert isinstance(node_type.scope, Scope)

        with err_desc("One and only one edge table must be defined", node.pos):
            assert len(children["edge"]) == 1
        edge_table = children["edge"][0]
        edge_type = scope.resolve("edge")
        assert isinstance(edge_type, BuiltinType)
        edge_type.scope = edge_table.scope

        distributions = [d for ds in children["distributions"] for d in ds]
        normal_dists = [d for d in distributions if isinstance(d, NormalDist)]
        uniform_dists = [d for d in distributions if isinstance(d, UniformDist)]
        discrete_dists = [d for d in distributions if isinstance(d, DiscreteDist)]

        contagions = children["contagion"]
        for contagion in contagions:
            node_type.scope.define(contagion.name, contagion)

        functions = children["function"]

        intervene = [f for f in functions if f.name == "intervene"]
        with err_desc("One and only one intervene function must be defined", node.pos):
            assert len(intervene) == 1
        intervene = intervene[0]

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
            pos=node.pos,
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
