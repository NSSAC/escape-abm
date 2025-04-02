"""Miscellaneous code."""

from __future__ import annotations

from typing import override, Any, Iterator

from pydantic import BaseModel, Field
import rich
import rich.markup
from rich.tree import Tree
from rich.pretty import Pretty


class SourcePosition(BaseModel):
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

    def get_bytes(self) -> bytes:
        return self.source_bytes[self.byte_range[0] : self.byte_range[1]]

    @property
    def text(self) -> str:
        return self.get_bytes().decode()


class RichException(Exception):
    """Exception with a rich_print method."""

    def rich_print(self):
        rich.print(f"[red]{self!s}[/red]")


class CodeError(RichException):
    """Error attributable to a section of the code."""

    def __init__(self, type: str, description: str, pos: SourcePosition | None):
        """
        Initialize.

        :param type: type of error.
        :param description: description of error.
        :param pos: position in source.
        """

        super().__init__()
        self.type = type
        self.description = description
        self.pos = pos

    def __str__(self):
        return f"{self.type}: {self.description}"

    @override
    def rich_print(self):
        if self.pos is not None:
            etype = f"[red]{self.type}[/red]"
            fpath = f"[yellow]{self.pos.source}[/yellow]"
            line = self.pos.line
            col = self.pos.col
            expl = rich.markup.escape(self.description)

            rich.print(f"{etype}:{fpath}:{line}:{col}: {expl}")
            print(self.pos.text)
        else:
            etype = f"[red]{self.type}[/red]"
            expl = rich.markup.escape(self.description)
            rich.print(f"{etype}: {expl}")


class CodeErrorList(RichException):
    """List of errors."""

    def __init__(self, description: str, errors: list[CodeError]):
        super().__init__()
        self.description = description
        self.errors = errors

    def __str__(self):
        return self.description

    @override
    def rich_print(self):
        rich.print(f"[red]{self}[/red]")
        for error in self.errors:
            error.rich_print()


class Scope(BaseModel):
    """Scope / Symbol table."""

    name: str
    names: dict[str, Any] = Field(default_factory=dict)
    parent: Scope | None = Field(default=None)
    children: list[Scope] = Field(default_factory=list)

    def model_post_init(self, _: Any) -> None:
        if self.parent is not None:
            self.parent.children.append(self)

    def __rich_repr__(self):
        yield "name", self.name
        yield "names", {k: type(v).__name__ for k, v in self.names.items()}
        if self.parent is not None:
            yield "parent", self.parent.name

    def resolve(self, k: str) -> Any:
        if k in self.names:
            return self.names[k]
        elif self.parent is not None:
            return self.parent.resolve(k)

        raise KeyError(f"{k} is not defined.")

    def define(self, k: str, v: Any):
        if k in self.names:
            raise ValueError(f"{k} has been already defined.")
        self.names[k] = v

    def undef(self, k: str):
        del self.names[k]

    def root(self) -> Scope:
        ret = self
        while ret.parent is not None:
            ret = ret.parent
        return ret

    def visit(self) -> Iterator[tuple[str, Any]]:
        yield from self.names.items()
        for child in self.children:
            yield from child.visit()

    def rich_tree(self, tree: Tree | None = None) -> Tree:
        if tree is None:
            tree = Tree(Pretty(self))
        else:
            tree = tree.add(Pretty(self))

        for child in self.children:
            child.rich_tree(tree)

        return tree
