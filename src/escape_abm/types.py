"""Scope and types."""

from __future__ import annotations

import networkx as nx
from typing import TypeGuard, Any
from dataclasses import dataclass, field

from .misc import TypeError


@dataclass
class Type:
    name: str
    symtab: dict[str, Any] = field(repr=False)


def make_type(name: str) -> Type:
    return Type(name=name, symtab=dict())


@dataclass
class FunctionType(Type):
    params: tuple[Type, ...]
    return_: Type


def make_function_type(params: tuple[Type, ...], return_: Type) -> FunctionType:
    name = [p.name for p in params]
    name = " x ".join(name)
    name = name + " -> " + return_.name
    return FunctionType(name=name, symtab=dict(), params=params, return_=return_)


_TYPE_GRAPH = nx.DiGraph()
_TYPE_OBJECT: dict[str, Type] = {}


def is_convertable_to(child: Type, ancestor: Type) -> bool:
    if child.name == ancestor.name:
        return True

    if child.name in _TYPE_GRAPH and ancestor.name in _TYPE_GRAPH:
        return nx.has_path(_TYPE_GRAPH, child.name, ancestor.name)
    else:
        return False


def lub_type(type1: Type, type2: Type) -> Type:
    if type1.name in _TYPE_GRAPH and type2.name in _TYPE_GRAPH:
        ltype_name = nx.lowest_common_ancestor(_TYPE_GRAPH, type1.name, type2.name)
        return _TYPE_OBJECT[ltype_name]
    else:
        return _TYPE_OBJECT["_type"]


def setup_type_system():
    _TYPE_GRAPH.clear()
    _TYPE_OBJECT.clear()

    nx.add_path(_TYPE_GRAPH, ("bool", "uint", "int", "float"))
    nx.add_path(_TYPE_GRAPH, ("i8", "i16", "i32", "i64", "int"))
    nx.add_path(_TYPE_GRAPH, ("u8", "u16", "u32", "u64", "uint"))
    nx.add_path(_TYPE_GRAPH, ("f32", "f64", "float"))

    nx.add_path(_TYPE_GRAPH, ("str", "_type"))
    nx.add_path(_TYPE_GRAPH, ("node", "_table", "_type"))
    nx.add_path(_TYPE_GRAPH, ("edge", "_table"))
    nx.add_path(_TYPE_GRAPH, ("void", "_type"))

    nx.add_path(_TYPE_GRAPH, ("_enum", "_type"))
    nx.add_path(_TYPE_GRAPH, ("_contagion", "_type"))

    for name in _TYPE_GRAPH.nodes:
        _TYPE_OBJECT[name] = make_type(name)


def add_enum_type(name: str) -> Type:
    if name in _TYPE_GRAPH:
        raise TypeError(f"Type '{name}' has already been defined")

    type = make_type(name)
    nx.add_path(_TYPE_GRAPH, (name, "_enum"))
    _TYPE_OBJECT[type.name] = type
    return type


def add_contagion_type(name: str) -> Type:
    if name in _TYPE_GRAPH:
        raise TypeError(f"Type '{name}' has already been defined")

    type = make_type(name)
    nx.add_path(_TYPE_GRAPH, (name, "_contagion"))
    _TYPE_OBJECT[type.name] = type
    return type


def get_type(name: str) -> Type:
    if name not in _TYPE_OBJECT:
        raise TypeError(f"Undefined type {name}")

    return _TYPE_OBJECT[name]


def is_numeric_type(type: Type) -> bool:
    return is_convertable_to(type, _TYPE_OBJECT["float"])


def is_integral_type(type: Type) -> bool:
    return is_convertable_to(type, _TYPE_OBJECT["int"])


def is_boolean_type(type: Type) -> bool:
    return type.name == "bool"


def is_enum_type(type: Type) -> bool:
    return is_convertable_to(type, _TYPE_OBJECT["_enum"])


def is_contagion_type(type: Type) -> bool:
    return is_convertable_to(type, _TYPE_OBJECT["_contagion"])


def is_function_type(type: Type) -> TypeGuard[FunctionType]:
    return isinstance(type, FunctionType)
