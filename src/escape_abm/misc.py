"""Miscellaneous code."""

from __future__ import annotations

from dataclasses import dataclass

import rich
import rich.markup


@dataclass
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

    def get_bytes(self) -> bytes:
        return self.source_bytes[self.byte_range[0] : self.byte_range[1]]

    @property
    def text(self) -> str:
        return self.get_bytes().decode()

    def __str__(self) -> str:
        return f"{self.source}:{self.line}:{self.col}"

    def __rich__(self) -> str:
        return f"[yellow]{self.source}[/yellow]:{self.line}:{self.col}"


class CodeError(Exception):
    """Error attributable to user code."""

    def __init__(self, type: str, description: str, pos: SourcePosition | None = None):
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

    def __str__(self) -> str:
        if self.pos is None:
            return f"{self.type}: {self.description}"
        else:
            return f"{self.type}:{self.pos}: {self.description}"

    def __rich__(self) -> str:
        if self.pos is None:
            return f"[red]{self.type}[/red]: {self.description}"
        else:
            return f"[red]{self.type}[/red]:{self.pos.__rich__()}: {rich.markup.escape(self.description)}"


class CodeErrorList(Exception):
    """List of errors attributable to user code."""

    def __init__(self, description: str, errors: list[CodeError]):
        super().__init__()
        self.description = description
        self.errors = errors

    def __str__(self) -> str:
        return self.description + "\n".join(str(e) for e in self.errors)

    def __rich__(self) -> str:
        ret = [f"[red]{self.description}[/red]"]
        for error in self.errors:
            ret.append(error.__rich__())
        return "\n".join(ret)
