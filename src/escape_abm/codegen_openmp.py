"""Codegen for CPU using OpenMP."""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path
from collections import defaultdict
from importlib.resources import files
from tempfile import TemporaryDirectory
from typing import assert_never, Any, TypeVar, ParamSpec, Callable

import click
import jinja2
import rich
import rich.markup
from pydantic import BaseModel
from typeguard import check_type, TypeCheckError

from .alias_table import AliasTable
from .misc import EslError, EslErrorList, SourcePosition, RichException
from .parse_tree import mk_pt
from .ast import mk_ast
from .check_ast import check_ast, is_node_set
from . import ast
from .click_helpers import (
    simulation_file_option,
    gen_code_dir_option,
    existing_gen_code_dir_option,
    existing_input_file_option,
    output_file_option,
)

Params = ParamSpec("Params")
Type = TypeVar("Type")


TEMPLATE_LOADER = jinja2.PackageLoader(
    package_name="escape_abm", package_path="templates"
)

ENVIRONMENT = jinja2.Environment(
    loader=TEMPLATE_LOADER,
    undefined=jinja2.StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
)


def render(template: str, **kwargs) -> str:
    return ENVIRONMENT.get_template(f"{template}.jinja2").render(**kwargs)


def register_filter(name: str):
    if name in ENVIRONMENT.filters:
        raise RuntimeError(f"Filter with name {name!r} has already been defined")

    def wrapper(wrapped: Callable[Params, str]) -> Callable[Params, str]:
        ENVIRONMENT.filters[name] = wrapped
        return wrapped

    return wrapper


STATIC_DIR = files("escape_abm.static")


def smallest_uint_type(max_val: int) -> str | None:
    if max_val < 2**8:
        return "u8"
    elif max_val < 2**16:
        return "u16"
    elif max_val < 2**32:
        return "u32"
    elif max_val < 2**64:
        return "u64"
    else:
        return None


def enum_base_type(t: ast.EnumType) -> str:
    type = smallest_uint_type(len(t.consts) - 1)
    if type is None:
        raise EslError("Large enumeration", "Enum too large", t.pos)
    return type


# fmt: off
TYPE_TO_CTYPE = {
    "int":   "int_type",
    "uint":  "uint_type",
    "float": "float_type",
    "bool":  "bool_type",
    "size":  "size_type",

    "u8":  "uint8_t",
    "u16": "uint16_t",
    "u32": "uint32_t",
    "u64": "uint64_t",

    "i8":  "int8_t",
    "i16": "int16_t",
    "i32": "int32_t",
    "i64": "int64_t",

    "f32": "float",
    "f64": "double",

    "node":  "node_index_type",
    "edge":  "edge_index_type",
}
# fmt: on

# fmt: off
BUILTIN_FN_TO_CFN = {
    "len": "len",
    "abs": "std::abs",
    "min": "std::fmin",
    "max": "std::fmax",
    "exp": "std::exp",
    "exp2": "std::exp2",
    "log": "std::log",
    "log2": "std::log2",
    "pow": "std::pow",
    "sin": "std::sin",
    "cos": "std::cos",
}
# fmt: on


@register_filter("mangle")
def mangle(*args: str) -> str:
    if len(args) == 1:
        return "_" + args[0]

    return "_" + "__".join(args)


def tref_str(x: ast.TValueRef) -> str:
    match x:
        case ast.BuiltinType():
            return TYPE_TO_CTYPE[x.name]
        case ast.EnumType():
            return mangle(x.name)
        case _ as unexpected:
            assert_never(unexpected)


def cref_str(x: ast.CValueRef) -> str:
    match x:
        case ast.BuiltinFunction():
            return BUILTIN_FN_TO_CFN[x.name]
        case ast.Function():
            return mangle(x.name)
        case ast.NormalDist():
            return mangle(x.name)
        case ast.UniformDist():
            return mangle(x.name)
        case ast.DiscreteDist():
            return mangle(x.name)
        case _ as unexpected:
            assert_never(unexpected)


def ref_str(x: ast.RValueRef) -> str:
    match x:
        case ast.BuiltinGlobal():
            return x.name
        case ast.EnumConstant():
            return mangle(x.name)
        case ast.Global():
            return mangle(x.name)
        case ast.Param():
            return mangle(x.name)
        case ast.Variable():
            return mangle(x.name)
        case ast.NodeSet():
            return mangle(x.name)
        case ast.BuiltinNodeset():
            return x.name
        case ast.EdgeSet():
            return mangle(x.name)
        case ast.BuiltinEdgeset():
            return x.name

        # Node / Edge fields
        case [ast.Param() | ast.Variable() as v, ast.NodeField() as f]:
            v_ref = ref_str(v)
            f_ref = mangle(f.name)
            return f"NODE_TABLE->{f_ref}[{v_ref}]"
        case [ast.Param() | ast.Variable() as e, ast.EdgeField() as f]:
            e_ref = ref_str(e)
            f_ref = mangle(f.name)
            return f"EDGE_TABLE->{f_ref}[{e_ref}]"

        case [
            ast.Param() | ast.Variable() as v,
            ast.Contagion() as c,
            ast.StateAccessor(),
        ]:
            v_ref = ref_str(v)
            f_ref = mangle(c.name) + "_state"
            return f"NODE_TABLE->{f_ref}[{v_ref}]"

        case [
            ast.Param() | ast.Variable() as e,
            ast.SourceNodeAccessor(),
        ]:
            e_ref = ref_str(e)
            return f"EDGE_TABLE->source_node_index[{e_ref}]"

        case [
            ast.Param() | ast.Variable() as e,
            ast.TargetNodeAccessor(),
        ]:
            e_ref = ref_str(e)
            return f"EDGE_TABLE->target_node_index[{e_ref}]"

        case [
            ast.Param() | ast.Variable() as e,
            ast.SourceNodeAccessor(),
            ast.NodeField() as f,
        ]:
            e_ref = ref_str(e)
            f_ref = mangle(f.name)
            return f"NODE_TABLE->{f_ref}[EDGE_TABLE->source_node_index[{e_ref}]]"

        case [
            ast.Param() | ast.Variable() as e,
            ast.TargetNodeAccessor(),
            ast.NodeField() as f,
        ]:
            e_ref = ref_str(e)
            f_ref = mangle(f.name)
            return f"NODE_TABLE->{f_ref}[EDGE_TABLE->target_node_index[{e_ref}]]"

        case [
            ast.Param() | ast.Variable() as e,
            ast.SourceNodeAccessor(),
            ast.Contagion() as c,
            ast.StateAccessor(),
        ]:
            e_ref = ref_str(e)
            f_ref = mangle(c.name) + "_state"
            return f"NODE_TABLE->{f_ref}[EDGE_TABLE->source_node_index[{e_ref}]]"

        case [
            ast.Param() | ast.Variable() as e,
            ast.TargetNodeAccessor(),
            ast.Contagion() as c,
            ast.StateAccessor(),
        ]:
            e_ref = ref_str(e)
            f_ref = mangle(c.name) + "_state"
            return f"NODE_TABLE->{f_ref}[EDGE_TABLE->target_node_index[{e_ref}]]"

        case _ as unexpected:
            assert_never(unexpected)


def cpp_operator(t: str) -> str:
    match t:
        case "or":
            return "||"
        case "and":
            return "&&"
        case "not":
            return "!"
        case _:
            return t


@register_filter("expression")
def expression_str(e: ast.Expression) -> str:
    match e:
        case bool():
            return str(int(e))
        case int():
            return str(e)
        case float():
            return str(e)
        case ast.UnaryExpression():
            op = cpp_operator(e.operator)
            eo = expression_str(e.argument)
            return f"{op}{eo}"
        case ast.BinaryExpression():
            left = expression_str(e.left)
            op = cpp_operator(e.operator)
            right = expression_str(e.right)
            return f"{left} {op} {right}"
        case ast.ParenthesizedExpression():
            eo = expression_str(e.expression)
            return f"({eo})"
        case ast.Reference():
            value = e.value
            try:
                check_type(value, ast.RValueRef)
            except TypeCheckError:
                raise EslError(
                    "Invalid reference", f"Can't get value of {e.name}", e.pos
                )
            return ref_str(value)
        case ast.TemplateVariable():
            if e.pos is None:
                line, col = 1, 1
            else:
                line, col = e.pos.line, e.pos.col
            return f"UNDEFINED_TEMPLATE_VARIABLE_{line}_{col}"
        case ast.FunctionCall():
            function = e.function.value
            try:
                check_type(function, ast.CValueRef)
            except TypeCheckError:
                raise EslError("Invalid callable", f"Can't call {function.name}", e.pos)

            function = cref_str(function)
            args = [expression_str(a) for a in e.args]
            args = ", ".join(args)
            return f"{function}({args})"
        case _ as unexpected:
            assert_never(unexpected)


@register_filter("fn_expression")
def fn_expression_str(e: ast.Expression, fn_param: str) -> str:
    match e:
        case ast.Reference() as r:
            match r.value:
                case ast.Function() as f:
                    return cref_str(f) + f"({ fn_param })"
                case ast.NormalDist() | ast.DiscreteDist() | ast.UniformDist() as d:
                    return cref_str(d) + "()"

    return expression_str(e)


@register_filter("typename")
def typename_str(x: ast.Reference | None) -> str:
    match x:
        case ast.Reference() as r:
            return tref_str(r.value)

    return "void"


@register_filter("line_pragma")
def line_pragma(pos: SourcePosition | None) -> str:
    match pos:
        case SourcePosition():
            return f'#line {pos.line} "{pos.source}"'

    return ""


def asdict(m: BaseModel) -> dict:
    return {k: getattr(m, k) for k in m.model_fields.keys()}


def is_all_set(r: ast.Reference) -> bool:
    match r.value:
        case ast.BuiltinNodeset(name="ALL_NODES"):
            return True
        case ast.BuiltinEdgeset(name="ALL_EDGES"):
            return True

    return False


@register_filter("statement")
def statement_str(s: ast.Statement) -> str:
    match s:
        case ast.PassStatement():
            return render("pass_statement", **asdict(s))
        case ast.ReturnStatement():
            return render("return_statement", **asdict(s))
        case ast.IfStatement():
            return render("if_statement", **asdict(s))
        case ast.SwitchStatement():
            return render("switch_statement", **asdict(s))
        case ast.WhileLoop():
            return render("while_loop", **asdict(s))
        case ast.CallStatement():
            return render("call_statement", **asdict(s))
        case ast.UpdateStatement():
            left = s.left.value
            try:
                check_type(left, ast.LValueRef)
            except TypeCheckError:
                raise EslError(
                    "Invalid reference", f"Can't update {s.left.name}", s.left.pos
                )
            left_str = ref_str(left)
            return render("update_statement", left_str=left_str, **asdict(s))
        case ast.PrintStatement():
            args = []
            for arg in s.args:
                if isinstance(arg, str):
                    args.append(arg)
                else:
                    arg = expression_str(arg)
                    arg = f"std::to_string({arg})"
                    args.append(arg)
            args = f" << {s.sep} << ".join(args)
            print_statement_str = "std::cout << " + args + f" << {s.end} << std::flush"
            return render(
                "print_statement",
                print_statement_str=print_statement_str,
                **asdict(s),
            )
        case ast.Variable():
            return render("variable", **asdict(s))
        case ast.SelectStatement():
            return render(
                "select_statement_launch_openmp",
                **asdict(s),
            )
        case ast.SampleStatement():
            is_all_sample = is_all_set(s.parent)
            return render(
                "sample_statement_launch_openmp",
                is_all_sample=is_all_sample,
                **asdict(s),
            )
        case ast.ApplyStatement():
            return render(
                "apply_statement_launch_openmp",
                **asdict(s),
            )
        case ast.ReduceStatement():
            is_node_reduce = is_node_set(s.set)
            is_all_reduce = is_all_set(s.set)
            return render(
                "reduce_statement_launch_openmp",
                is_node_reduce=is_node_reduce,
                is_all_reduce=is_all_reduce,
                **asdict(s),
            )
        case _ as unexpected:
            assert_never(unexpected)


@register_filter("enum_defn")
def enum_defn_str(x: ast.EnumType) -> str:
    base_type = TYPE_TO_CTYPE[enum_base_type(x)]
    return render("enum_defn", base_type=base_type, **asdict(x))


@register_filter("global_defn")
def global_defn_str(x: ast.Global) -> str:
    return render("global_defn", **asdict(x))


@register_filter("function_decl")
def function_decl_str(x: ast.Function) -> str:
    return render("function_decl", **asdict(x))


@register_filter("function_defn")
def function_defn_str(x: ast.Function) -> str:
    return render("function_defn", variables=x.variables(), **asdict(x))


@register_filter("node_table_defn")
def node_table_defn_str(x: ast.NodeTable) -> str:
    return render("node_table_defn", **asdict(x))


@register_filter("edge_table_defn")
def edge_table_defn_str(x: ast.EdgeTable) -> str:
    return render("edge_table_defn", **asdict(x))


@register_filter("normal_dist_defn")
def normal_dist_defn_str(x: ast.NormalDist) -> str:
    return render("normal_dist_defn", **asdict(x))


@register_filter("uniform_dist_defn")
def uniform_dist_defn_str(x: ast.UniformDist) -> str:
    return render("uniform_dist_defn", **asdict(x))


@register_filter("discrete_dist_defn")
def discrete_dist_defn_str(x: ast.DiscreteDist) -> str:
    table = AliasTable.make(x.ps)
    probs = [str(p) for p in table.probs]
    alias = [str(p) for p in table.alias]
    return render("discrete_dist_defn", probs=probs, alias=alias, **asdict(x))


@register_filter("select_statement_defn")
def select_statement_defn_str(x: ast.SelectStatement) -> str:
    return render("select_statement_defn_openmp", **asdict(x))


@register_filter("apply_statement_defn")
def apply_statement_defn_str(x: ast.ApplyStatement) -> str:
    is_all_apply = is_all_set(x.set)
    is_node_apply = is_node_set(x.set)
    return render(
        "apply_statement_defn_openmp",
        is_all_apply=is_all_apply,
        is_node_apply=is_node_apply,
        **asdict(x),
    )


@register_filter("reduce_statement_defn")
def reduce_statement_defn_str(x: ast.ReduceStatement) -> str:
    is_all_reduce = is_all_set(x.set)
    is_node_reduce = is_node_set(x.set)
    return render(
        "reduce_statement_defn_openmp",
        is_all_reduce=is_all_reduce,
        is_node_reduce=is_node_reduce,
        **asdict(x),
    )


class SingleExitTransition(BaseModel):
    entry: str
    exit: str
    dwell: ast.Expression


class MultiExitTransition(BaseModel):
    entry: str
    exits: list[str]
    pexprs: list[ast.Expression]
    dwells: list[ast.Expression]
    num_exits: int


@register_filter("contagion_methods")
def contagion_methods_str(x: ast.Contagion) -> str:
    num_states = len(x.state_type.value.consts)

    entry_transitions = defaultdict(list)
    for t in x.transitions:
        entry_transitions[t.entry.value.name].append(t)

    single_exit_transitions: list[SingleExitTransition] = []
    multi_exit_transitions: list[MultiExitTransition] = []
    for vs in entry_transitions.values():
        if len(vs) == 1:
            single_exit_transitions.append(
                SingleExitTransition(
                    entry=vs[0].entry.value.name,
                    exit=vs[0].exit.value.name,
                    dwell=vs[0].dwell,
                )
            )
        else:
            multi_exit_transitions.append(
                MultiExitTransition(
                    entry=vs[0].entry.value.name,
                    exits=[v.exit.value.name for v in vs],
                    pexprs=[v.pexpr for v in vs],
                    dwells=[v.dwell for v in vs],
                    num_exits=len(vs),
                )
            )

    # entry -> [exit -> [contact]]
    grouped_transmissions = defaultdict(lambda: defaultdict(list))

    for t in x.transmissions:
        contact = t.contact.value.name
        entry = t.entry.value.name
        exit = t.exit.value.name
        grouped_transmissions[entry][exit].append(contact)

    grouped_transmissions = [
        (entry, [(exit, contacts) for exit, contacts in xs.items()])
        for entry, xs in grouped_transmissions.items()
    ]

    return render(
        "contagion_methods",
        num_states=num_states,
        single_exit_transitions=single_exit_transitions,
        multi_exit_transitions=multi_exit_transitions,
        grouped_transmissions=grouped_transmissions,
        **asdict(x),
    )


def simulator_str(source: ast.Source) -> str:
    try:
        return render("simulator_openmp", **asdict(source))
    except EslError:
        raise
    except jinja2.TemplateError:
        raise
    except Exception as e:
        estr = f"{e.__class__.__name__}: {e!s}"
        raise EslError(
            "Codegen Error", f"Failed to generate code: ({estr})", source.pos
        )


def do_prepare(gen_src_dir: Path, input: Path, source: ast.Source) -> None:
    if source.template_variables:
        errors: list[EslError] = []
        for tvar in source.template_variables:
            e = EslError("Codegen error", "Unrendered template variable", tvar.pos)
            errors.append(e)
        raise EslErrorList(
            "Can't run prepare for code with unrendered template variables.", errors
        )

    gen_src_dir.mkdir(parents=True, exist_ok=True)

    with open(gen_src_dir / "CMakeLists.txt", "wt") as fobj:
        fobj.write(
            render(
                "cmakelists_txt_openmp",
                module=source.module,
                input=str(input.absolute()),
            )
        )

    cmd = f"cmake -S '{gen_src_dir!s}' -B '{gen_src_dir!s}/build'"
    cmd = shlex.split(cmd)
    subprocess.run(cmd, check=True)


def do_compile(build_dir: Path, source: ast.Source) -> None:
    with open(build_dir / "simulator_openmp.cpp", "wt") as fobj:
        fobj.write(simulator_str(source))

    with open(build_dir / "simulation_common_openmp.h", "wt") as fobj:
        fobj.write(STATIC_DIR.joinpath("simulation_common_openmp.h").read_text())


def do_build(gen_src_dir: Path) -> None:
    cmd = f"cmake --build '{gen_src_dir!s}/build'"
    cmd = shlex.split(cmd)
    subprocess.run(cmd, check=True)


def do_simulate(
    gen_src_dir: Path,
    input_file: Path,
    output_file: Path,
    num_ticks: int,
    configs: dict[str, Any],
    verbose: bool = False,
) -> None:
    assert (gen_src_dir / "build/simulator_openmp").exists(), "Simulator doesn't exist"
    assert input_file.exists(), "Input file doesn't exist"

    env = dict(os.environ)

    env["OMP_PROC_BIND"] = "true"

    env["INPUT_FILE"] = str(input_file)
    env["OUTPUT_FILE"] = str(output_file)
    env["NUM_TICKS"] = str(num_ticks)
    for key, value in configs.items():
        if isinstance(value, bool):
            value = int(value)
        env[key.upper()] = str(value)

    cmd = f"'{gen_src_dir!s}/build/simulator_openmp'"
    cmd = shlex.split(cmd)
    if verbose:
        subprocess.run(cmd, env=env, check=True)
    else:
        subprocess.run(
            cmd,
            env=env,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )


@click.group()
def codegen_openmp():
    """Generate parallel simulator using OpenMP."""


@codegen_openmp.command()
@gen_code_dir_option
@simulation_file_option
def prepare(gen_code_dir: Path, simulation_file: Path):
    """Generate CMakeLists.txt and run cmake."""
    if not gen_code_dir.exists():
        gen_code_dir.mkdir(parents=True, exist_ok=True)

    try:
        pt = mk_pt(str(simulation_file), simulation_file.read_bytes())
        ast = mk_ast(simulation_file, pt)
        check_ast(ast)
        do_prepare(gen_code_dir, simulation_file, ast)
    except RichException as e:
        e.rich_print()
        raise SystemExit(1)


@codegen_openmp.command()
@existing_gen_code_dir_option
@simulation_file_option
def compile(gen_code_dir: Path, simulation_file: Path):
    """Compile simulator code."""
    try:
        pt = mk_pt(str(simulation_file), simulation_file.read_bytes())
        ast = mk_ast(simulation_file, pt)
        check_ast(ast)
        do_compile(gen_code_dir, ast)
    except RichException as e:
        e.rich_print()
        raise SystemExit(1)


@codegen_openmp.command()
@click.option(
    "-n",
    "--num-ticks",
    type=int,
    default=0,
    show_default=True,
    help="Number of ticks.",
)
@existing_input_file_option
@output_file_option
@simulation_file_option
def run(num_ticks: int, output_file: Path, input_file: Path, simulation_file: Path):
    """Build and run simulator."""
    try:
        with TemporaryDirectory(
            prefix=f"{simulation_file.stem}-", suffix="-escape_abm"
        ) as temp_output_dir:
            gen_src_dir = Path(temp_output_dir).absolute()
            rich.print(f"[cyan]Temp output dir:[/cyan] {gen_src_dir!s}")

            pt = mk_pt(str(simulation_file), simulation_file.read_bytes())
            ast = mk_ast(simulation_file, pt)

            rich.print("[cyan]Preparing ...[/cyan]")
            do_prepare(gen_src_dir, simulation_file, ast)

            rich.print("[cyan]Building ...[/cyan]")
            do_build(gen_src_dir)

            rich.print("[cyan]Simulating ...[/cyan]")
            do_simulate(
                gen_src_dir,
                input_file=input_file,
                output_file=output_file,
                num_ticks=num_ticks,
                configs={},
                verbose=True,
            )
    except RichException as e:
        e.rich_print()
        raise SystemExit(1)
