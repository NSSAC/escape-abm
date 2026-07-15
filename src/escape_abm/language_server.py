"""Lnaguage server."""

import logging
from functools import cache
from pathlib import Path

import click
import tree_sitter as ts
from lsprotocol import types
from pygls.lsp.server import LanguageServer
from platformdirs import user_log_dir

from .misc import SourcePosition, EslError, EslErrorList
from .tree_sitter_bindings import get_parser, enum_query
from .parse_tree import mk_pt
from .ast import mk_ast
from .check_ast import check_ast
from .codegen_openmp import simulator_str

server = LanguageServer("esl-server", "v0.1")

_CURRENT_COMPLETIONS: list[types.CompletionItem] = []


def error_to_diagnostic(
    type: str, explanation: str, pos: SourcePosition | None
) -> types.Diagnostic:
    if pos is None:
        range = types.Range(
            start=types.Position(0, 0),
            end=types.Position(0, 0),
        )
    else:
        range = types.Range(
            start=types.Position(pos.start[0], pos.start[1]),
            end=types.Position(pos.end[0], pos.end[1]),
        )
    return types.Diagnostic(
        range=range,
        severity=types.DiagnosticSeverity.Error,
        message=f"{type}: {explanation}",
        source="esl-server",
    )


# fmt: off
BUILTIN_TYPE_DOCS = [
    # Integer types
    ("int", "Integer"),
    ("i8", "8 bit integer"),
    ("i16", "16 bit integer"),
    ("i32", "32 bit integer"),
    ("i64", "64 bit integer"),

    # Unsigned integer types
    ("uint", "Unsigned integer"),
    ("u8", "8 bit unsigned integer"),
    ("u16", "16 bit unsigned integer"),
    ("u32", "32 bit unsigned integer"),
    ("u64", "64 bit unsigned integer"),

    # Floating point types
    ("float", "Floating point type"),
    ("f32", "32 bit floating point type"),
    ("f64", "64 bit floating point type"),

    # Other types (that users are expected to write)
    ("size", "Size type"),
    ("bool", "Boolean type"),
]

BUILTIN_CONSTANT_DOCS = [
    ("NUM_TICKS", "int", "Number of ticks"),
    ("CUR_TICK", "int", "Current tick"),
    ("NUM_NODES", "size", "Number of nodes"),
    ("NUM_EDGES", "size", "Number of edges"),
]

BUILTIN_FUNCTION_DOCS = [
    ("len", "len(nodeset | edgeset) : size", "Size of a nodeset or edgeset"),
]
# fmt: on


@cache
def builtin_completions() -> list[types.CompletionItem]:
    completions: list[types.CompletionItem] = []
    for type, doc in BUILTIN_TYPE_DOCS:
        completions.append(
            types.CompletionItem(
                label=type,
                kind=types.CompletionItemKind.Class,
                detail="builtin type",
                documentation=doc,
            )
        )

    for const, detail, doc in BUILTIN_CONSTANT_DOCS:
        completions.append(
            types.CompletionItem(
                label=const,
                kind=types.CompletionItemKind.Constant,
                detail=detail,
                documentation=doc,
            )
        )

    for func, detail, doc in BUILTIN_FUNCTION_DOCS:
        completions.append(
            types.CompletionItem(
                label=func,
                kind=types.CompletionItemKind.Function,
                detail=detail,
                documentation=doc,
            )
        )

    return completions


def enum_completions(root_node: ts.Node) -> list[types.CompletionItem]:
    completions: list[types.CompletionItem] = []
    for capture, nodes in ts.QueryCursor(enum_query()).captures(root_node).items():
        for node in nodes:
            match capture:
                case "enum_name":
                    completions.append(
                        types.CompletionItem(
                            label=node.text.decode(),
                            kind=types.CompletionItemKind.Enum,
                            detail="enumeration type",
                        )
                    )
                case "enum_const":
                    completions.append(
                        types.CompletionItem(
                            label=node.text.decode(),
                            kind=types.CompletionItemKind.EnumMember,
                            detail="enumeration constant",
                        )
                    )

    return completions


def update_state(file_bytes: bytes | None):
    _CURRENT_COMPLETIONS.clear()
    _CURRENT_COMPLETIONS.extend(builtin_completions())

    if file_bytes is not None:
        parse_tree = get_parser().parse(file_bytes)
        _CURRENT_COMPLETIONS.extend(enum_completions(parse_tree.root_node))


@server.feature(
    types.TEXT_DOCUMENT_COMPLETION, types.CompletionOptions(trigger_characters=[","])
)
def completions(params: types.CompletionParams | None = None) -> types.CompletionList:
    """Returns completion items."""
    return types.CompletionList(is_incomplete=False, items=_CURRENT_COMPLETIONS)


@server.feature(types.TEXT_DOCUMENT_DID_OPEN)
@server.feature(types.TEXT_DOCUMENT_DID_SAVE)
@server.feature(types.TEXT_DOCUMENT_DID_CHANGE)
async def validate(
    ls: LanguageServer,
    params: (
        types.DidOpenTextDocumentParams
        | types.DidSaveTextDocumentParams
        | types.DidChangeTextDocumentParams
    ),
):
    global _MOST_RECENT_PARSE_TREE

    ls.window_log_message(types.LogMessageParams(types.MessageType.Info, "checking document"))

    file_path = Path(params.text_document.uri)
    doc = ls.workspace.get_text_document(params.text_document.uri)
    file_bytes = doc.source.encode()
    update_state(file_bytes)

    try:
        pt = mk_pt(str(file_path), file_bytes)
        ast = mk_ast(file_path, pt)
        check_ast(ast)
        simulator_str(ast)
    except EslErrorList as es:
        diagnostics = []
        for e in es.errors:
            diagnostic = error_to_diagnostic(e.type, e.description, e.pos)
            diagnostics.append(diagnostic)
        ls.text_document_publish_diagnostics(types.PublishDiagnosticsParams(params.text_document.uri, diagnostics))
        return
    except EslError as e:
        diagnostic = error_to_diagnostic(e.type, e.description, e.pos)
        ls.text_document_publish_diagnostics(types.PublishDiagnosticsParams(params.text_document.uri, [diagnostic]))
        return
    except Exception as e:
        diagnostic = error_to_diagnostic(
            "Language server error",
            f"Unexpected exception: type={type(e)!s}: {e!s}",
            None,
        )
        ls.text_document_publish_diagnostics(types.PublishDiagnosticsParams(params.text_document.uri, [diagnostic]))
        return

    # If we have no errors, remove all diagnostics
    ls.text_document_publish_diagnostics(types.PublishDiagnosticsParams(params.text_document.uri, []))


@click.group()
def language_server():
    """ESL language server."""


@language_server.command()
def io_server():
    """Start standard IO based server."""
    log_dir = user_log_dir(appname="esl-language-server", version="0.1")
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / "language-server.log"

    logging.basicConfig(filename=str(log_file), filemode="a", level=logging.INFO)

    update_state(None)
    server.start_io()


@language_server.command()
@click.option(
    "--host", default="127.0.0.1", show_default=True, help="Hostname to bind to."
)
@click.option(
    "--port", default=2087, show_default=True, type=int, help="Port to bind to."
)
def tcp_server(host: str, port: int):
    """Start TCP/IP based server."""
    logging.basicConfig(filemode="a", level=logging.INFO)

    update_state(None)
    server.start_tcp(host, port)
