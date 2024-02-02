"""Lnaguage server."""

import logging
from pathlib import Path

import click
from lsprotocol.types import (
    DidOpenTextDocumentParams,
    DidSaveTextDocumentParams,
    TEXT_DOCUMENT_DID_OPEN,
    TEXT_DOCUMENT_DID_SAVE,
    Diagnostic,
    DiagnosticSeverity,
    Range,
    Position,
)
from pygls.server import LanguageServer
from platformdirs import user_log_dir

from .parse_tree import mk_pt, ParseTreeConstructionError, SourcePosition
from .ast1 import mk_ast1, ASTConstructionError
from .codegen_cpu import Source as SourceCPU, CodegenError

server = LanguageServer("esl37-server", "v0.1")


def error_to_diagnostic(
    type: str, explanation: str, pos: SourcePosition | None
) -> Diagnostic:
    if pos is None:
        range = Range(
            start=Position(0, 0),
            end=Position(0, 0),
        )
    else:
        range = Range(
            start=Position(pos.start[0], pos.start[1]),
            end=Position(pos.end[0], pos.end[1]),
        )
    return Diagnostic(
        range=range,
        severity=DiagnosticSeverity.Error,
        message=f"{type}: {explanation}",
        source="esl37-server",
    )


@server.feature(TEXT_DOCUMENT_DID_OPEN)
@server.feature(TEXT_DOCUMENT_DID_SAVE)
async def make_diagnostics(
    ls: LanguageServer, params: DidOpenTextDocumentParams | DidSaveTextDocumentParams
):
    ls.show_message_log("checking document")

    file_path = Path(params.text_document.uri)
    doc = ls.workspace.get_document(params.text_document.uri)
    file_bytes = doc.source.encode()

    try:
        pt = mk_pt(str(file_path), file_bytes)
        ast1 = mk_ast1(file_path, pt)
        _ = SourceCPU.make(ast1)
    except ParseTreeConstructionError as e:
        diagnostics = []
        for error in e.errors:
            diagnostic = error_to_diagnostic(error.type, error.explanation, error.pos)
            diagnostics.append(diagnostic)
        ls.publish_diagnostics(params.text_document.uri, diagnostics)
        return
    except (ASTConstructionError, CodegenError) as e:
        diagnostic = error_to_diagnostic(e.type, e.explanation, e.pos)
        ls.publish_diagnostics(params.text_document.uri, [diagnostic])
        return

    # If we have no errors, remove all diagnostics
    ls.publish_diagnostics(params.text_document.uri, [])


@click.group()
def language_server():
    """ESL37 language server."""


@language_server.command()
def io_server():
    """Start standard IO based server."""
    log_dir = user_log_dir(appname="esl37-language-server", version="0.1")
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / "language-server.log"

    logging.basicConfig(filename=str(log_file), filemode="a", level=logging.INFO)

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

    server.start_tcp(host, port)
