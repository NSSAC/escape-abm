"""Python bindings for the tree-sitter-parser."""

from functools import cache
from tree_sitter import Language, Parser, Query

import tree_sitter_esl as esl


@cache
def get_language() -> Language:
    return Language(esl.language())


@cache
def get_parser() -> Parser:
    parser = Parser(get_language())
    return parser


@cache
def enum_query() -> Query:
    query = """
    (enum
        name: (identifier) @enum_name
        const: (identifier) @enum_const
     )
    """
    language = get_language()
    return language.query(query)
