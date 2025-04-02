"""Scope and types."""

from __future__ import annotations

import networkx as nx
from functools import cached_property
from pydantic import BaseModel, Field
from typing import Self

from .misc import Scope


class BuiltinType(BaseModel):
    name: str
    width: int | None = Field(default=None, repr=False)
    scope: Scope | None = Field(default=None, repr=False)


class EnumType(BaseModel):
    name: str
    consts: list[str]


_EXISTENTIAL_TYPE_COUNTER: int = 0


class ExistentialType(BaseModel):
    index: int
    parent: Type | None = None

    @cached_property
    def name(self) -> str:
        if self.parent is not None:
            return self.parent.name
        else:
            return f"unknown_type:{self.index}"

    @classmethod
    def create(cls) -> Self:
        global _EXISTENTIAL_TYPE_COUNTER
        obj = cls(index=_EXISTENTIAL_TYPE_COUNTER)
        _EXISTENTIAL_TYPE_COUNTER += 1
        return obj

    @staticmethod
    def reset_counter():
        global _EXISTENTIAL_TYPE_COUNTER
        _EXISTENTIAL_TYPE_COUNTER = 0


class FunctionType(BaseModel):
    params: list[Type]
    return_: Type

    @property
    def name(self) -> str:
        ps = [p.name for p in self.params]
        ps = " x ".join(ps)
        ret = self.return_
        return f"{ps} -> {ret}"


Type = BuiltinType | EnumType | ExistentialType


def make_type_graph() -> nx.DiGraph:
    G = nx.DiGraph()
    nx.add_path(G, ("bool", "uint", "int", "float", "type"))
    nx.add_path(G, ("node", "type"))
    nx.add_path(G, ("edge", "type"))
    nx.add_path(G, ("str", "type"))
    nx.add_path(G, ("void", "type"))
    return G


def is_sub_type(child: str, ancestor: str, G: nx.DiGraph) -> bool:
    if child == ancestor:
        return True

    return nx.has_path(G, child, ancestor)


def lub_type(type1: str, type2: str, G: nx.DiGraph) -> str:
    return nx.lowest_common_ancestor(G, type1, type2)


def add_base_types(scope: Scope):
    scope.define("int", BuiltinType(name="int"))
    scope.define("i8", BuiltinType(name="int", width=8))
    scope.define("i16", BuiltinType(name="int", width=16))
    scope.define("i32", BuiltinType(name="int", width=32))
    scope.define("i64", BuiltinType(name="int", width=64))

    scope.define("uint", BuiltinType(name="uint"))
    scope.define("u8", BuiltinType(name="uint", width=8))
    scope.define("u16", BuiltinType(name="uint", width=16))
    scope.define("u32", BuiltinType(name="uint", width=32))
    scope.define("u64", BuiltinType(name="uint", width=64))

    scope.define("float", BuiltinType(name="float"))
    scope.define("f32", BuiltinType(name="float", width=32))
    scope.define("f64", BuiltinType(name="float", width=64))

    scope.define("bool", BuiltinType(name="bool"))
    scope.define("str", BuiltinType(name="str"))

    scope.define("node", BuiltinType(name="node"))
    scope.define("edge", BuiltinType(name="edge"))

    scope.define("_void", BuiltinType(name="void"))
