"""Builtin functions."""

from dataclasses import dataclass

from .scope import get_scope
from .types import Type, get_type, make_fn_type


@dataclass
class BuiltinFunction:
    name: str
    type: Type


@dataclass
class BuiltinGlobalVariable:
    name: str
    type: Type


_BUILTIN_VARIABLES = [
    ["NUM_TICKS", "int"],
    ["CUR_TICK", "int"],
    ["NUM_NODES", "int"],
    ["NUM_EDGES", "int"],
]


_BUILTIN_FUNCTIONS = [
    ["sample_uniform", ["float", "float"], "float"],
    ["sample_normal", ["float", "float"], "float"],
    ["abs", ["float"], "float"],
    ["min", ["float", "float"], "float"],
    ["max", ["float", "float"], "float"],
    ["exp", ["float"], "float"],
    ["exp2", ["float"], "float"],
    ["log", ["float"], "float"],
    ["log2", ["float"], "float"],
    ["pow", ["float", "float"], "float"],
    ["sin", ["float"], "float"],
    ["cos", ["float"], "float"],
]


def define_builtin_variables():
    scope = get_scope()
    for name, type in _BUILTIN_VARIABLES:
        scope.define(name, BuiltinGlobalVariable(name=name, type=get_type(type)))


def define_builtin_functions():
    scope = get_scope()
    for name, params, return_ in _BUILTIN_FUNCTIONS:
        fn_type = make_fn_type(params, return_)
        scope.define(name, BuiltinFunction(name=name, type=fn_type))
