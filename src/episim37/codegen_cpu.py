"""Codegen for CPU."""

from pathlib import Path
from typing import Callable, TypeVar, Any

import shlex
from textwrap import dedent
from subprocess import run as do_run
from tempfile import TemporaryDirectory

import click
import attrs
import jinja2

import rich
from rich.syntax import Syntax

from .parse_tree import mk_pt, ParseTreeConstructionError
from .ast1 import mk_ast1, ASTConstructionError

T = TypeVar("T")


ENVIRONMENT = jinja2.Environment(
    loader=jinja2.PackageLoader(package_name="episim37", package_path="."),
    undefined=jinja2.StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
)

CODEGEN_TEMPLATES: dict[str, jinja2.Template] = {}
CODEGEN_FUNCTIONS: dict[str, Callable[[Any], str]] = {}


def register_template(cls: type, macro_name: str):
    global CODEGEN_TEMPLATES

    cls_name = cls.__name__

    import_stmt = "{%% from 'codegen_cpu.jinja2' import %s %%}" % macro_name
    macro_stmt = "{{ %s(o) }}" % macro_name
    template = "\n".join([import_stmt, macro_stmt])
    template = ENVIRONMENT.from_string(template)

    CODEGEN_TEMPLATES[cls_name] = template


def register_function(cls: type, callable: Callable[[Any], str]):
    global CODEGEN_FUNCTIONS

    cls_name = cls.__name__
    CODEGEN_FUNCTIONS[cls_name] = callable


def codegen(o: Any) -> str:
    cls_name = o.__class__.__name__

    if cls_name in CODEGEN_FUNCTIONS:
        callable = CODEGEN_FUNCTIONS[cls_name]
        return callable(o)

    if cls_name in CODEGEN_TEMPLATES:
        template = CODEGEN_TEMPLATES[cls_name]
        return template.render(o=o)

    return f"/* MISSING CODEGEN {cls_name} */"


ENVIRONMENT.filters["codegen"] = codegen

from .ast1 import Source, PrintStatement

register_function(str, str)
register_function(int, str)
register_function(float, str)
register_function(bool, lambda x: "true" if x else "false")

register_template(Source, "source")
register_template(PrintStatement, "print_statement")


@attrs.define
class CmakeLists:
    source: Source


register_template(CmakeLists, "cmake_lists")


def do_build(output: Path | None, input: Path, file_bytes: bytes) -> Path:
    pt = mk_pt(file_bytes)
    source = mk_ast1(input, pt)

    if output is None:
        output = Path.cwd() / source.module
    output.mkdir(parents=True, exist_ok=True)

    with open(output / "simulator.cpp", "wt") as fobj:
        fobj.write(codegen(source))

    cmake_lists = CmakeLists(source)
    with open(output / "CMakeLists.txt", "wt") as fobj:
        fobj.write(codegen(cmake_lists))

    return output


@click.group()
def cpu():
    """Generate code targetted for single CPU."""


@cpu.command()
@click.option(
    "-o",
    "--output",
    type=click.Path(exists=False, file_okay=False, dir_okay=True, path_type=Path),
    help="Output directory",
)
@click.argument(
    "input",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
)
def build(output: Path | None, input: Path):
    """Build simulator."""
    file_bytes = input.read_bytes()
    try:
        do_build(output, input, file_bytes)
    except (ParseTreeConstructionError, ASTConstructionError) as e:
        e.rich_print(input, file_bytes)


def pprint_cmd(cmd):
    cmd = dedent(cmd).strip()
    cmd = Syntax(cmd, "bash", theme="gruvbox-dark")
    rich.print(cmd)


def verbose_run(cmd, *args, **kwargs):
    pprint_cmd(cmd)
    cmd = shlex.split(cmd)
    return do_run(cmd, *args, **kwargs)


@cpu.command()
@click.argument(
    "input",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
)
def run(input: Path):
    """Build and run simulator."""
    file_bytes = input.read_bytes()
    try:
        with TemporaryDirectory(
            prefix=f"{input.stem}-", suffix="-episim37"
        ) as temp_output_dir:
            output = Path(temp_output_dir).absolute()
            rich.print(f"[cyan]Output dir:[/cyan] {output!s}")

            rich.print("[cyan]Creating simulator[/cyan]")
            output = do_build(output, input, file_bytes)

            rich.print("[cyan]Running cmake[/cyan]")
            cmd = "cmake ."
            verbose_run(cmd, cwd=output)

            rich.print("[cyan]Running make[/cyan]")
            cmd = "make"
            verbose_run(cmd, cwd=output)

            rich.print("[cyan]Running simulator[/cyan]")
            cmd = "./simulator"
            verbose_run(cmd, cwd=output)
    except (ParseTreeConstructionError, ASTConstructionError) as e:
        e.rich_print(input, file_bytes)
