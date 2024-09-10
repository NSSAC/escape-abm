"""Helper utilities for creating/reading input.h5 file."""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass

import numpy as np
import polars as pl
import h5py as h5

import click
import rich

from .misc import RichException
from .parse_tree import mk_pt
from .ast import BuiltinType, EdgeTable, NodeTable, mk_ast, EnumType, Source
from .codegen_openmp import enum_base_type
from .output_helpers import save_df
from .click_helpers import (
    simulation_file_option,
    existing_node_file_option,
    existing_edge_file_option,
    input_file_option,
    existing_input_file_option,
    node_file_option,
    edge_file_option,
)

# fmt: off
TYPE_TO_DTYPE = {
    "int": np.int64,
    "uint": np.uint64,
    "float": np.float32,
    "bool": np.uint8,
    "size": np.int64,

    "u8":  np.uint8,
    "u16": np.uint16,
    "u32": np.uint32,
    "u64": np.uint64,

    "i8":  np.int8,
    "i16": np.int16,
    "i32": np.int32,
    "i64": np.int64,

    "f32":  np.float32,
    "f64": np.float64,

    "node": np.int32,
    "edge": np.int64,
}
# fmt: on


@dataclass
class NodeTableMeta:
    columns: list[str]
    enum_dtypes: dict[str, pl.Enum]
    out_dtypes: dict[str, type]
    key: str

    @classmethod
    def from_source(cls, node_table: NodeTable) -> NodeTableMeta:
        columns = []
        enum_dtypes = {}
        out_dtypes = {}

        for field in node_table.fields:
            if not field.is_static:
                continue

            columns.append(field.name)
            match field.type.value:
                case EnumType() as t:
                    enum_dtypes[field.name] = pl.Enum(t.consts)
                    out_dtypes[field.name] = TYPE_TO_DTYPE[enum_base_type(t)]
                case BuiltinType() as t:
                    out_dtypes[field.name] = TYPE_TO_DTYPE[t.name]
                case unexpected:
                    raise RuntimeError(f"Expected type: {unexpected!r}")

        key = node_table.key.name
        return cls(columns, enum_dtypes, out_dtypes, key)


@dataclass
class EdgeTableMeta:
    columns: list[str]
    enum_dtypes: dict[str, pl.Enum]
    out_dtypes: dict[str, type]
    target_node_key: str
    source_node_key: str

    @classmethod
    def from_source(cls, edge_table: EdgeTable) -> EdgeTableMeta:
        columns = []
        enum_dtypes = {}
        out_dtypes = {}

        for field in edge_table.fields:
            if not field.is_static:
                continue

            columns.append(field.name)
            match field.type.value:
                case EnumType() as t:
                    enum_dtypes[field.name] = pl.Enum(t.consts)
                    out_dtypes[field.name] = TYPE_TO_DTYPE[enum_base_type(t)]
                case BuiltinType() as t:
                    out_dtypes[field.name] = TYPE_TO_DTYPE[t.name]
                case unexpected:
                    raise RuntimeError(f"Expected type: {unexpected!r}")

        out_dtypes["_source_node_index"] = TYPE_TO_DTYPE["node"]
        out_dtypes["_target_node_index"] = TYPE_TO_DTYPE["node"]

        target_node_key = edge_table.target_node_key.name
        source_node_key = edge_table.source_node_key.name

        return cls(
            columns,
            enum_dtypes,
            out_dtypes,
            target_node_key,
            source_node_key,
        )


def read_table(file: Path, columns: list[str]) -> pl.DataFrame:
    if file.suffix == ".csv":
        return pl.read_csv(file, columns=columns)
    elif file.suffix == ".parquet":
        return pl.read_parquet(file, columns=columns)
    else:
        rich.print("[red]Unknown filetype for node file[/red]")
        raise SystemExit(1)


def check_null(table: pl.DataFrame, name: str):
    null_count = table.null_count()
    for col in table.columns:
        if null_count[col][0] > 0:
            rich.print(f"[red]Column {col} in {name} has null values.[/red]")
            raise SystemExit(1)


def make_source(simulation_file: Path) -> Source:
    simulation_bytes = simulation_file.read_bytes()
    pt = mk_pt(str(simulation_file), simulation_bytes)
    ast = mk_ast(simulation_file, pt)
    return ast


def make_node_table(node_file: Path, ntm: NodeTableMeta) -> pl.DataFrame:
    node_table = read_table(node_file, ntm.columns)
    for col, dtype in ntm.enum_dtypes.items():
        node_table = node_table.with_columns(node_table[col].cast(dtype).to_physical())
    check_null(node_table, "node table")
    return node_table


def make_edge_table(edge_file: Path, etm: EdgeTableMeta, key_idx: dict) -> pl.DataFrame:
    edge_table = read_table(edge_file, etm.columns)
    for col, dtype in etm.enum_dtypes.items():
        edge_table = edge_table.with_columns(edge_table[col].cast(dtype).to_physical())
    check_null(edge_table, "edge table")

    edge_table = edge_table.with_columns(
        edge_table[etm.target_node_key]
        .replace_strict(key_idx)
        .alias("_target_node_index"),
        edge_table[etm.source_node_key]
        .replace_strict(key_idx)
        .alias("_source_node_index"),
    )
    return edge_table


def make_in_inc_csr_indptr(edge_table: pl.DataFrame, Vn: int) -> np.ndarray:
    target_node_index = edge_table["_target_node_index"].to_numpy()
    edge_count = np.bincount(target_node_index, minlength=Vn)
    cum_edge_count = np.cumsum(edge_count)
    in_inc_csr_indptr = np.zeros(Vn + 1, dtype=TYPE_TO_DTYPE["edge"])
    in_inc_csr_indptr[1:] = cum_edge_count
    return in_inc_csr_indptr


def make_input_file(
    data_file: Path,
    node_table: pl.DataFrame,
    edge_table: pl.DataFrame,
    ntm: NodeTableMeta,
    etm: EdgeTableMeta,
    in_inc_csr_indptr: np.ndarray,
):
    size_dtype = TYPE_TO_DTYPE["size"]

    with h5.File(data_file, "w") as fobj:
        for col, dtype in ntm.out_dtypes.items():
            data = node_table[col].to_numpy()
            data = np.asarray(data, dtype)
            fobj.create_dataset(f"/node/{col}", data=data)

        for col, dtype in etm.out_dtypes.items():
            data = edge_table[col].to_numpy()
            data = np.asarray(data, dtype)
            fobj.create_dataset(f"/edge/{col}", data=data)

        fobj.create_dataset("/incoming/incidence/csr/indptr", data=in_inc_csr_indptr)

        fobj.attrs.create("num_nodes", len(node_table), dtype=size_dtype)
        fobj.attrs.create("num_edges", len(edge_table), dtype=size_dtype)


def do_read_nodes_df(data_file: Path, ntm: NodeTableMeta) -> pl.DataFrame:
    df = {}
    with h5.File(data_file, "r") as fobj:
        for col in ntm.columns:
            s = fobj[f"/node/{col}"][...]  # type: ignore
            s = pl.Series(s)
            df[col] = s

    for col, dtype in ntm.enum_dtypes.items():
        df[col] = df[col].cast(dtype)

    return pl.DataFrame(df)


def read_nodes_df(simulation_file: Path, input_file: Path) -> pl.DataFrame:
    source = make_source(simulation_file)
    ntm = NodeTableMeta.from_source(source.node_table)
    return do_read_nodes_df(input_file, ntm)


def do_read_edges_df(data_file: Path, etm: EdgeTableMeta) -> pl.DataFrame:
    df = {}
    with h5.File(data_file, "r") as fobj:
        for col in etm.columns + ["_target_node_index", "_source_node_index"]:
            s = fobj[f"/edge/{col}"][...]  # type: ignore
            s = pl.Series(s)
            df[col] = s

    for col, dtype in etm.enum_dtypes.items():
        df[col] = df[col].cast(dtype)

    return pl.DataFrame(df)


def read_edges_df(simulation_file: Path, input_file: Path) -> pl.DataFrame:
    source = make_source(simulation_file)
    etm = EdgeTableMeta.from_source(source.edge_table)
    return do_read_edges_df(input_file, etm)


def do_prepare_input(
    ntm: NodeTableMeta,
    etm: EdgeTableMeta,
    node_file: Path,
    edge_file: Path,
    input_file: Path,
) -> None:
    # Step 1: Preprocess the node table files
    rich.print("[cyan]Reading node table.[/cyan]")
    node_table = make_node_table(node_file, ntm)

    # Step 2: Sort the node table dataset
    rich.print("[cyan]Sorting node table.[/cyan]")
    node_table = node_table.sort(ntm.key)

    # Step 3: Create a node index
    rich.print("[cyan]Making node index.[/cyan]")
    key_idx = {k: idx for idx, k in enumerate(node_table[ntm.key])}

    Vn = len(node_table)
    print("### num_nodes: ", Vn)

    # Step 4: Preprocess the edge table files
    rich.print("[cyan]Reading edge table.[/cyan]")
    edge_table = make_edge_table(edge_file, etm, key_idx)

    # Step 5: Sort the edge table dataset
    rich.print("[cyan]Sorting edge table.[/cyan]")
    edge_table = edge_table.sort(["_target_node_index", "_source_node_index"])

    # Step 6: Make incoming incidence CSR graph's indptr
    rich.print("[cyan]Computing incoming incidence CSR graph's indptr.[/cyan]")
    in_inc_csr_indptr = make_in_inc_csr_indptr(edge_table, Vn)

    En = len(edge_table)
    print("### num_edges: ", En)

    # Step 7: Create the data file.
    rich.print("[cyan]Creating input file.[/cyan]")
    make_input_file(input_file, node_table, edge_table, ntm, etm, in_inc_csr_indptr)

    rich.print("[green]Input file created successfully.[/green]")


@click.command()
@simulation_file_option
@existing_node_file_option
@existing_edge_file_option
@input_file_option
def prepare_input(
    simulation_file: Path, node_file: Path, edge_file: Path, input_file: Path
):
    """Prepare input for simulation."""
    try:
        rich.print("[cyan]Parsing simulator code.[/cyan]")
        source = make_source(simulation_file)
        ntm = NodeTableMeta.from_source(source.node_table)
        etm = EdgeTableMeta.from_source(source.edge_table)

        do_prepare_input(ntm, etm, node_file, edge_file, input_file)
    except RichException as e:
        e.rich_print()
        raise SystemExit(1)


@click.group
def process_input():
    """Process input file."""


@process_input.command()
@simulation_file_option
@existing_input_file_option
@node_file_option
def extract_nodes(
    simulation_file: Path,
    input_file: Path,
    node_file: Path,
):
    """Extract node table from simulation input."""
    try:
        df = read_nodes_df(simulation_file, input_file)
        save_df(df, node_file)
    except RichException as e:
        e.rich_print()
        raise SystemExit(1)


@process_input.command()
@simulation_file_option
@existing_input_file_option
@edge_file_option
def extract_edges(
    simulation_file: Path,
    input_file: Path,
    edge_file: Path,
):
    """Extract edge table from simulation input."""
    try:
        df = read_edges_df(simulation_file, input_file)
        save_df(df, edge_file)
    except RichException as e:
        e.rich_print()
        raise SystemExit(1)
