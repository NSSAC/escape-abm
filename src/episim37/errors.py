"""Error types."""

from enum import Enum
from pathlib import Path
from contextlib import contextmanager
from typing import Generator

import rich
import rich.markup
import attrs
from tree_sitter import Node


class ErrorType(Enum):
    ParseError = "Failed to parse"
    MissingToken = "Missing token"


@attrs.define
class Error:
    type: ErrorType
    explanation: str
    node: Node


PARSE_ERRORS: list[Error] | None = None


@contextmanager
def capture_errors() -> Generator[list[Error], None, None]:
    global PARSE_ERRORS

    prev_errors = PARSE_ERRORS
    PARSE_ERRORS = []
    try:
        yield PARSE_ERRORS
    finally:
        PARSE_ERRORS = prev_errors


def append_error(type: ErrorType, explanation: str, node: Node) -> None:
    global PARSE_ERRORS

    if PARSE_ERRORS is None:
        return

    error = Error(type=type, explanation=explanation, node=node)
    PARSE_ERRORS.append(error)


def print_errors(errors: list[Error], file_bytes: bytes, file_path: Path):
    fpath = str(file_path)
    for error in errors:
        etype = error.type.value
        line = error.node.start_point[0] + 1
        col = error.node.start_point[1] + 1
        expl = rich.markup.escape(error.explanation)

        rich.print(f"[yellow]{etype}[/yellow]: {fpath}:{line}:{col}: {expl}")
        match error.type:
            case ErrorType.ParseError:
                error_part = file_bytes[
                    error.node.start_byte : error.node.end_byte
                ].decode()
                print(error_part)
                print("")
