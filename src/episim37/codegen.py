"""Codegen for CPU."""

from typing import Callable, TypeVar, Any
from pathlib import Path

import click
import jinja2

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

    import_stmt = "{%% from 'codegen.jinja2' import %s %%}" % macro_name
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
register_function(bool, lambda x: 'true' if x else 'false')

register_template(Source, "source")
register_template(PrintStatement, "print_statement")


@click.command()
@click.argument(
    "input",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
)
def compile(input: Path):
    """Generate code targetted for single CPU."""
    file_bytes = input.read_bytes()
    try:
        pt = mk_pt(file_bytes)
        ast1 = mk_ast1(input, pt)
        print(codegen(ast1))
    except (ParseTreeConstructionError, ASTConstructionError) as e:
        e.rich_print(input, file_bytes)
