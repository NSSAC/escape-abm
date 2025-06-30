"""Scope and types."""

from __future__ import annotations

import networkx as nx
from typing import Iterable, TypeGuard
from dataclasses import dataclass

from .misc import CodeError


@dataclass
class BuiltinType:
    name: str


@dataclass
class EnumType:
    name: str
    consts: tuple[str, ...]


@dataclass
class FunctionType:
    params: tuple[Type, ...]
    return_: Type

    @property
    def name(self) -> str:
        ps = [p.name for p in self.params]
        ps = " x ".join(ps)
        ret = self.return_
        return f"{ps} -> {ret}"


Type = BuiltinType | EnumType | FunctionType
UserDefinedType = EnumType

_TYPE_GRAPH = nx.DiGraph()
_TYPE_OBJECT: dict[str, Type] = {}


def is_convertable_to(child: Type, ancestor: Type) -> bool:
    if child.name == ancestor.name:
        return True

    if child.name in _TYPE_GRAPH and ancestor.name in _TYPE_GRAPH:
        return nx.has_path(_TYPE_GRAPH, child.name, ancestor.name)
    else:
        return False


def lub_type(type1: Type, type2: Type) -> Type | None:
    if type1.name in _TYPE_GRAPH and type2.name in _TYPE_GRAPH:
        ltype_name = nx.lowest_common_ancestor(_TYPE_GRAPH, type1.name, type2.name)
        return _TYPE_OBJECT.get(ltype_name, None)
    else:
        return None


def add_user_defined_type(type: UserDefinedType):
    if type.name in _TYPE_GRAPH:
        raise CodeError(
            "Type redefinition", f"Type '{type.name}' has already been defined"
        )

    _TYPE_GRAPH.add_node(type.name)
    _TYPE_OBJECT[type.name] = type


def setup_type_system():
    _TYPE_GRAPH.clear()
    _TYPE_OBJECT.clear()

    nx.add_path(_TYPE_GRAPH, ("bool", "uint", "int", "float"))
    nx.add_path(_TYPE_GRAPH, ("i8", "i16", "i32", "i64", "int"))
    nx.add_path(_TYPE_GRAPH, ("u8", "u16", "u32", "u64", "uint"))
    nx.add_path(_TYPE_GRAPH, ("f32", "f64", "float"))

    _TYPE_GRAPH.add_node("str")
    _TYPE_GRAPH.add_node("void")
    _TYPE_GRAPH.add_node("node")
    _TYPE_GRAPH.add_node("edge")

    for type_name in _TYPE_GRAPH.nodes:
        _TYPE_OBJECT[type_name] = BuiltinType(type_name)


def visible_types() -> Iterable[Type]:
    for name, type in _TYPE_OBJECT.items():
        if not name.startswith("_"):
            yield type


def get_type(name: str) -> Type:
    if name not in _TYPE_OBJECT:
        raise CodeError("Type error", f"Undefined type {name}")

    return _TYPE_OBJECT[name]


def is_numeric_type(type: Type) -> bool:
    return is_convertable_to(type, _TYPE_OBJECT["float"])


def is_integral_type(type: Type) -> bool:
    return is_convertable_to(type, _TYPE_OBJECT["int"])


def is_boolean_type(type: Type) -> bool:
    return type.name == "bool"


def is_enum_type(type: Type) -> TypeGuard[EnumType]:
    return isinstance(type, EnumType)


def is_function_type(type: Type) -> TypeGuard[FunctionType]:
    return isinstance(type, FunctionType)
