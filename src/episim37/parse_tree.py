"""Parse using tree sitter."""

from __future__ import annotations

from pathlib import Path

import click
from tree_sitter import Node as TSNode

import rich
from rich.tree import Tree
from rich.pretty import Pretty

from .tree_sitter_bindings import get_parser
from .errors import ErrorType, append_error


def check_parse_errors(node: TSNode):
    if node.type == "ERROR":
        append_error(type=ErrorType.ParseError, explanation=node.sexp(), node=node)
    elif node.is_missing:
        append_error(type=ErrorType.MissingToken, explanation=node.type, node=node)
    elif node.has_error:
        for child in node.children:
            check_parse_errors(child)


class PTNode:
    def __init__(self, node: TSNode):
        self.node = node
        self.type = self.node.type
        self.is_named = node.is_named
        self.is_missing = node.is_missing

    @property
    def text(self) -> str:
        return self.node.text.decode()

    def field(self, name: str) -> PTNode | None:
        child = self.node.child_by_field_name(name)
        if child is None:
            return None
        else:
            return self.__class__(child)

    def fields(self, name: str) -> list[PTNode]:
        children = self.node.children_by_field_name(name)
        children = [self.__class__(child) for child in children if child.is_named]
        return children

    @property
    def children(self) -> list[PTNode]:
        return [self.__class__(child) for child in self.node.children]

    @property
    def named_children(self) -> list[PTNode]:
        return [self.__class__(child) for child in self.node.named_children]

    def __rich_repr__(self):
        yield "type", self.type
        yield "is_missing", self.is_missing, False
        yield "is_named", self.is_named, True
        if not self.node.child_count:
            yield "text", self.text

    def rich_tree(self, tree: Tree | None = None) -> Tree:
        if tree is None:
            tree = Tree(Pretty(self))
        else:
            tree = tree.add(Pretty(self))

        for child in self.named_children:
            child.rich_tree(tree)

        return tree


def mk_pt(file_bytes: bytes) -> PTNode:
    """Make parse tree."""
    parser = get_parser()
    parse_tree = parser.parse(file_bytes)
    check_parse_errors(parse_tree.root_node)
    root_node = PTNode(parse_tree.root_node)
    return root_node


@click.command()
@click.argument(
    "filename",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
)
def print_parse_tree(filename: Path):
    """Print the parse tree."""

    file_bytes = filename.read_bytes()
    root_node = mk_pt(file_bytes)
    rich.print(root_node.rich_tree())
