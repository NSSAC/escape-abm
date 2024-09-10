"""Miscellaneous code."""

from __future__ import annotations

from pydantic import BaseModel
import rich
import rich.markup


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


class EslError(RichException):
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


class EslErrorList(RichException):
    """List of errors."""

    def __init__(self, description: str, errors: list[EslError]):
        super().__init__()
        self.description = description
        self.errors = errors

    def __str__(self):
        return self.description

    def rich_print(self):
        rich.print(f"[red]{self}[/red]")
        for error in self.errors:
            error.rich_print()
