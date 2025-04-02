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


def get_query(q: str) -> Query:
    language = get_language()
    return language.query(q)
