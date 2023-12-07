"""Codegen for CPU."""

from pathlib import Path

import shlex
from textwrap import dedent
from subprocess import run as do_run
from tempfile import TemporaryDirectory

import click
import jinja2

import rich
from rich.syntax import Syntax

from .parse_tree import mk_pt, ParseTreeConstructionError
from .ast1 import mk_ast1, ASTConstructionError
from .ir1 import mk_ir1, IRConstructionError, Source

ENVIRONMENT = jinja2.Environment(
    loader=jinja2.PackageLoader(package_name="episim37", package_path="."),
    undefined=jinja2.StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
)


def gen_simulator_cpp(source: Source) -> str:
    template = (
        "{% from 'codegen_cpu.jinja2' import simulator_cpp %}"
        "{{ simulator_cpp(source) }}"
    )
    template = ENVIRONMENT.from_string(template)
    return template.render(source=source)


def gen_cmake_lists(source: Source) -> str:
    template = (
        "{% from 'codegen_cpu.jinja2' import cmake_lists %}" "{{ cmake_lists(source) }}"
    )
    template = ENVIRONMENT.from_string(template)
    return template.render(source=source)


def do_build(
    output: Path | None, input: Path, node_idx_type: str, edge_idx_type: str
) -> Path:
    input_bytes = input.read_bytes()
    pt = mk_pt(str(input), input_bytes)
    ast1 = mk_ast1(input, pt)
    ir1 = mk_ir1(ast1, node_idx_type, edge_idx_type)

    if output is None:
        output = input.parent
    output.mkdir(parents=True, exist_ok=True)

    with open(output / "simulator.cpp", "wt") as fobj:
        fobj.write(gen_simulator_cpp(ir1))

    with open(output / "CMakeLists.txt", "wt") as fobj:
        fobj.write(gen_cmake_lists(ir1))

    return output


@click.group()
def codegen_cpu():
    """Generate code targetted for single CPU."""


@codegen_cpu.command()
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
    node_idx_type = "uint32_t"
    edge_idx_type = "uint64_t"

    try:
        do_build(output, input, node_idx_type, edge_idx_type)
    except (ParseTreeConstructionError, ASTConstructionError, IRConstructionError) as e:
        e.rich_print()


def pprint_cmd(cmd):
    cmd = dedent(cmd).strip()
    cmd = Syntax(cmd, "bash", theme="gruvbox-dark")
    rich.print(cmd)


def verbose_run(cmd, *args, **kwargs):
    pprint_cmd(cmd)
    cmd = shlex.split(cmd)
    return do_run(cmd, *args, **kwargs)


@codegen_cpu.command()
@click.argument(
    "input",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
)
def run(input: Path):
    """Build and run simulator."""
    node_idx_type = "uint32_t"
    edge_idx_type = "uint64_t"

    try:
        with TemporaryDirectory(
            prefix=f"{input.stem}-", suffix="-episim37"
        ) as temp_output_dir:
            output = Path(temp_output_dir).absolute()
            rich.print(f"[cyan]Temp output dir:[/cyan] {output!s}")

            rich.print("[cyan]Creating simulator[/cyan]")
            output = do_build(output, input, node_idx_type, edge_idx_type)

            rich.print("[cyan]Running cmake[/cyan]")
            cmd = "cmake ."
            verbose_run(cmd, cwd=output)

            rich.print("[cyan]Running make[/cyan]")
            cmd = "make"
            verbose_run(cmd, cwd=output)

            rich.print("[cyan]Running simulator[/cyan]")
            cmd = "./simulator"
            verbose_run(cmd, cwd=output)
    except (ParseTreeConstructionError, ASTConstructionError, IRConstructionError) as e:
        e.rich_print()
