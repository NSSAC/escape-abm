"""Parse using tree sitter."""

from __future__ import annotations

from pathlib import Path

import click
from tree_sitter import Node as TSNode

import rich
import rich.markup
from rich.tree import Tree
from rich.pretty import Pretty

from .misc import SourcePosition, EslError, EslErrorList
from .tree_sitter_bindings import get_parser
from .click_helpers import simulation_file_option


def tsnode_to_pos(node: TSNode, source: str, source_bytes: bytes) -> SourcePosition:
    start = (node.start_point[0], node.start_point[1])
    end = (node.end_point[0], node.end_point[1])
    byte_range = (node.start_byte, node.end_byte)
    return SourcePosition(
        source=source,
        source_bytes=source_bytes,
        start=start,
        end=end,
        byte_range=byte_range,
    )


FILTERED_NODES = ["comment", "template_block"]


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
        return tsnode_to_pos(self.node, self.source, self.source_bytes)

    def field(self, name: str) -> PTNode:
        child = self.node.child_by_field_name(name)
        if child is None:
            raise KeyError(
                f"Node of type '{self.type}' doesn't have a field named '{name}'"
            )
        return PTNode(self.source, self.source_bytes, child)

    def maybe_field(self, name: str) -> PTNode | None:
        child = self.node.child_by_field_name(name)
        if child is None:
            return None
        else:
            return PTNode(self.source, self.source_bytes, child)

    def fields(self, name: str) -> list[PTNode]:
        children = self.node.children_by_field_name(name)
        children = [
            PTNode(self.source, self.source_bytes, child)
            for child in children
            if child.is_named
        ]
        return children

    @property
    def children(self) -> list[PTNode]:
        return [
            PTNode(self.source, self.source_bytes, child)
            for child in self.node.children
            if child.type not in FILTERED_NODES
        ]

    @property
    def named_children(self) -> list[PTNode]:
        return [
            PTNode(self.source, self.source_bytes, child)
            for child in self.node.named_children
            if child.type not in FILTERED_NODES
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


def check_parse_errors(
    source: str, source_bytes: bytes, node: TSNode, errors: list[EslError] | None = None
) -> list[EslError]:
    if errors is None:
        errors = []

    if node.type == "ERROR":
        errors.append(
            EslError(
                "Parse error",
                "Failed to parse",
                tsnode_to_pos(node, source, source_bytes),
            )
        )
    elif node.is_missing:
        errors.append(
            EslError(
                "Missing token",
                f"Expected token of type {node.type}",
                tsnode_to_pos(node, source, source_bytes),
            )
        )
    elif node.has_error:
        for child in node.children:
            check_parse_errors(source, source_bytes, child, errors)

    return errors


def mk_pt(source: str, source_bytes: bytes) -> PTNode:
    """Make parse tree."""
    parser = get_parser()
    parse_tree = parser.parse(source_bytes)
    root_node = parse_tree.root_node
    errors = check_parse_errors(source, source_bytes, root_node)
    if errors:
        raise EslErrorList("Failed to construct parse tree", errors)
    root_node = PTNode(source, source_bytes, root_node)
    return root_node


@click.command()
@simulation_file_option
def print_parse_tree(simulation_file: Path):
    """Print the parse tree."""
    file_bytes = simulation_file.read_bytes()
    try:
        pt = mk_pt(str(simulation_file), file_bytes)
        rich.print(pt.rich_tree())
    except EslErrorList as e:
        e.rich_print()
        raise SystemExit(1)
