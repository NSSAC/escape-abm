"""Program scope."""

from __future__ import annotations

from typing import Any, Iterator, overload
from dataclasses import dataclass, field

from rich.tree import Tree
from rich.pretty import Pretty

from .misc import CodeError


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

    @overload
    def resolve[T](self, k: str, type: type[T]) -> T: ...

    @overload
    def resolve[T](self, k: str, type: tuple[type[T], ...]) -> T: ...

    @overload
    def resolve(self, k: str, type: Any | None) -> Any: ...

    @overload
    def resolve(self, k: str) -> Any: ...

    def resolve(self, k: str, type=None):
        if k in self.names:
            ret = self.names[k]
        elif self.parent is not None:
            ret = self.parent.resolve(k, type)  # type: ignore
        else:
            raise CodeError("Scope error", f"{k} is not defined.")

        if type is None:
            return ret

        if not isinstance(ret, type):
            raise CodeError(
                "Type error", f"Expected object of type {type}; got {type(ret)}"
            )

        return ret

    def define(self, k: str, v: Any):
        if k in self.names:
            raise CodeError("Redefinition error", f"{k} has been already defined.")
        self.names[k] = v

    def undef(self, k: str):
        del self.names[k]

    def is_defined(self, k: str) -> bool:
        if k in self.names:
            return True
        elif self.parent is not None:
            return self.parent.is_defined(k)
        else:
            return False

    def clear(self):
        self.names.clear()
        for child in self.children:
            child.clear()
        self.children.clear()

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
