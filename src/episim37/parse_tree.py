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


@attrs.define
class SourcePosition:
    """Position in source."""

    source: str
    source_bytes: bytes
    start: tuple[int, int]  # line, col: zero indexed
    end: tuple[int, int]  # line, col: zero indexed
    byte_range: tuple[int, int]  # zero indexed

    @property
    def line(self):
        return self.start[0] + 1

    @property
    def col(self):
        return self.start[1] + 1

    @property
    def bytes(self) -> bytes:
        return self.source_bytes[self.byte_range[0] : self.byte_range[1]]

    @property
    def text(self) -> str:
        return self.bytes.decode()

    @classmethod
    def from_tsnode(
        cls, source: str, source_bytes: bytes, node: TSNode
    ) -> SourcePosition:
        start = (node.start_point[0], node.start_point[1])
        end = (node.end_point[0], node.end_point[1])
        byte_range = (node.start_byte, node.end_byte)
        return cls(source, source_bytes, start, end, byte_range)


class PTNode:
    """Parse tree node."""

    def __init__(self, source: str, source_bytes: bytes, node: TSNode):
        self.source = source
        self.source_bytes = source_bytes
        self.node = node
        self.type = self.node.type
        self.is_named = node.is_named
        self.is_missing = node.is_missing

    @property
    def text(self) -> str:
        return self.node.text.decode()

    @property
    def pos(self) -> SourcePosition:
        return SourcePosition.from_tsnode(self.source, self.source_bytes, self.node)

    def field(self, name: str) -> PTNode:
        child = self.node.child_by_field_name(name)
        if child is None:
            raise ValueError(
                f"Node of type '{self.type}' doesn't have a field named '{name}'"
            )
        return self.__class__(self.source, self.source_bytes, child)

    def maybe_field(self, name: str) -> PTNode | None:
        child = self.node.child_by_field_name(name)
        if child is None:
            return None
        else:
            return self.__class__(self.source, self.source_bytes, child)

    def fields(self, name: str) -> list[PTNode]:
        children = self.node.children_by_field_name(name)
        children = [
            self.__class__(self.source, self.source_bytes, child)
            for child in children
            if child.is_named
        ]
        return children

    @property
    def children(self) -> list[PTNode]:
        return [
            self.__class__(self.source, self.source_bytes, child)
            for child in self.node.children
        ]

    @property
    def named_children(self) -> list[PTNode]:
        return [
            self.__class__(self.source, self.source_bytes, child)
            for child in self.node.named_children
        ]

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
    pos: SourcePosition

    def rich_print(self):
        etype = f"[red]{self.type}[/red]"
        fpath = f"[yellow]{self.pos.source}[/yellow]"
        line = self.pos.line
        col = self.pos.col
        expl = rich.markup.escape(self.explanation)

        rich.print(f"{etype}:{fpath}:{line}:{col}:{expl}")
        if self.show_source:
            print(self.pos.text)


def check_parse_errors(
    source: str, source_bytes: bytes, node: TSNode, errors: list[Error] | None = None
) -> list[Error]:
    if errors is None:
        errors = []

    if node.type == "ERROR":
        errors.append(
            Error(
                "Parse error",
                node.sexp(),
                True,
                SourcePosition.from_tsnode(source, source_bytes, node),
            )
        )
    elif node.is_missing:
        errors.append(
            Error(
                "Missing token",
                node.type,
                True,
                SourcePosition.from_tsnode(source, source_bytes, node),
            )
        )
    elif node.has_error:
        for child in node.children:
            check_parse_errors(source, source_bytes, child, errors)

    return errors


class ParseTreeConstructionError(Exception):
    def __init__(self, errors: list[Error]):
        super().__init__()
        self.errors = errors

    def __str__(self):
        return "Failed to construct parse tree"

    def rich_print(self):
        rich.print(f"[red]{self}[/red]")
        for error in self.errors:
            error.rich_print()


def mk_pt(source: str, source_bytes: bytes) -> PTNode:
    """Make parse tree."""
    parser = get_parser()
    parse_tree = parser.parse(source_bytes)
    root_node = parse_tree.root_node
    errors = check_parse_errors(source, source_bytes, root_node)
    if errors:
        raise ParseTreeConstructionError(errors)
    root_node = PTNode(source, source_bytes, root_node)
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
        pt = mk_pt(str(filename), file_bytes)
        rich.print(pt.rich_tree())
    except ParseTreeConstructionError as e:
        e.rich_print()
