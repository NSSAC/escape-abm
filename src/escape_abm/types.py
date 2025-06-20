"""Scope and types."""

from __future__ import annotations

import networkx as nx
from dataclasses import dataclass, field

from .scope import Scope
from . import environ


@dataclass
class BuiltinType:
    name: str
    scope: Scope | None = field(default=None, repr=False)


@dataclass
class EnumType:
    name: str
    consts: list[str]


@dataclass
class FunctionType:
    params: list[DataType]
    return_: DataType

    @property
    def name(self) -> str:
        ps = [p.name for p in self.params]
        ps = " x ".join(ps)
        ret = self.return_
        return f"{ps} -> {ret}"


DataType = BuiltinType | EnumType


def is_sub_type(child: str, ancestor: str, type_graph: nx.DiGraph) -> bool:
    if child == ancestor:
        return True

    return nx.has_path(type_graph, child, ancestor)


def lub_type(type1: str, type2: str, type_graph: nx.DiGraph) -> str | None:
    return nx.lowest_common_ancestor(type_graph, type1, type2)


def do_setup_base_types():
    type_graph = nx.DiGraph()

    nx.add_path(type_graph, ("bool", "uint", "int", "float"))
    nx.add_path(type_graph, ("i8", "i16", "i32", "i64", "int"))
    nx.add_path(type_graph, ("u8", "u16", "u32", "u64", "uint"))
    nx.add_path(type_graph, ("f32", "f64", "float"))

    type_graph.add_node("node")
    type_graph.add_node("edge")
    type_graph.add_node("str")

    type_graph.add_node("_void")
    type_graph.add_node("_cdist")

    environ.set("type_graph", type_graph)

    scope = environ.get("global_scope")
    assert isinstance(scope, Scope)

    scope.define("int", BuiltinType(name="int"))
    scope.define("i8", BuiltinType(name="i8"))
    scope.define("i16", BuiltinType(name="i16"))
    scope.define("i32", BuiltinType(name="i32"))
    scope.define("i64", BuiltinType(name="i64"))

    scope.define("uint", BuiltinType(name="uint"))
    scope.define("u8", BuiltinType(name="u8"))
    scope.define("u16", BuiltinType(name="u16"))
    scope.define("u32", BuiltinType(name="u32"))
    scope.define("u64", BuiltinType(name="u64"))

    scope.define("float", BuiltinType(name="float"))
    scope.define("f32", BuiltinType(name="f32"))
    scope.define("f64", BuiltinType(name="f64"))

    scope.define("bool", BuiltinType(name="bool"))
    scope.define("str", BuiltinType(name="str"))

    scope.define("node", BuiltinType(name="node"))
    scope.define("edge", BuiltinType(name="edge"))

    scope.define("_void", BuiltinType(name="_void"))
    scope.define("_cdist", BuiltinType(name="_cdist"))
