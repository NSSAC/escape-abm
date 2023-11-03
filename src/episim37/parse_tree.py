"""Parse using tree sitter."""

from __future__ import annotations

from pathlib import Path

import click
import attrs
from tree_sitter import Node as TSNode

import rich
import rich.markup
from rich.tree import Tree
from rich.pretty import Pretty

from .tree_sitter_bindings import get_parser


class PTNode:
    def __init__(self, node: TSNode):
        self.node = node
        self.type = self.node.type
        self.is_named = node.is_named
        self.is_missing = node.is_missing

    @property
    def text(self) -> str:
        return self.node.text.decode()

    def field(self, name: str) -> PTNode:
        child = self.node.child_by_field_name(name)
        if child is None:
            raise ValueError(
                f"Node of type '{self.type}' doesn't have a field named '{name}'"
            )
        return self.__class__(child)

    def maybe_field(self, name: str) -> PTNode | None:
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


@attrs.define
class Error:
    type: str
    explanation: str
    show_source: bool
    node: TSNode

    def rich_print(self, file_path: Path, file_bytes: bytes):
        fpath = str(file_path)
        etype = self.type
        line = self.node.start_point[0] + 1
        col = self.node.start_point[1] + 1
        expl = rich.markup.escape(self.explanation)

        rich.print(f"[red]{etype}[/red]:[yellow]{fpath}[/yellow]:{line}:{col}: {expl}")
        if self.show_source:
            start = self.node.start_byte
            end = self.node.end_byte
            error_part = file_bytes[start:end].decode()
            print(error_part)


def check_parse_errors(node: TSNode, errors: list[Error] | None = None) -> list[Error]:
    if errors is None:
        errors = []

    if node.type == "ERROR":
        errors.append(Error("Parse error", node.sexp(), True, node))
    elif node.is_missing:
        errors.append(Error("Missing token", node.type, True, node))
    elif node.has_error:
        for child in node.children:
            check_parse_errors(child, errors)

    return errors


class ParseTreeConstructionError(Exception):
    def __init__(self, errors: list[Error]):
        super().__init__()
        self.errors = errors

    def __str__(self):
        return "Failed to construct parse tree"

    def rich_print(self, file_path: Path, file_bytes: bytes):
        rich.print(f"[red]{self}[/red]")
        for error in self.errors:
            error.rich_print(file_path, file_bytes)


def mk_pt(file_bytes: bytes) -> PTNode:
    """Make parse tree."""
    parser = get_parser()
    parse_tree = parser.parse(file_bytes)
    root_node = parse_tree.root_node
    errors = check_parse_errors(root_node)
    if errors:
        raise ParseTreeConstructionError(errors)
    root_node = PTNode(root_node)
    return root_node


@click.command()
@click.argument(
    "filename",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
)
def print_parse_tree(filename: Path):
    """Print the parse tree."""
    file_bytes = filename.read_bytes()
    try:
        pt = mk_pt(file_bytes)
        rich.print(pt.rich_tree())
    except ParseTreeConstructionError as e:
        e.rich_print(filename, file_bytes)
