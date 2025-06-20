"""Program scope."""

from __future__ import annotations

from typing import Any, Iterator
from dataclasses import dataclass, field

from rich.tree import Tree
from rich.pretty import Pretty


@dataclass
class Scope:
    """Progrma scope."""

    name: str
    names: dict[str, Any] = field(default_factory=dict)
    parent: Scope | None = field(default=None)
    children: list[Scope] = field(default_factory=list)

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
