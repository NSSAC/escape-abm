"""Helper utilities for creating/reading input.h5 file."""

from __future__ import annotations

from pathlib import Path

import attrs
import numpy as np
import polars as pl
import h5py as h5

import click
import rich

from .parse_tree import mk_pt, ParseTreeConstructionError
from .ast1 import mk_ast1, ASTConstructionError
from .codegen_cpu import (
    CodegenError,
    INC_CSR_INDPTR_DATASET_NAME,
    EnumType as Enum,
    Source as SourceCPU,
    DEFERRED_TYPES,
)
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
CTYPE_TO_DTYPE = {
    "uint8_t":  np.uint8,
    "uint16_t": np.uint16,
    "uint32_t": np.uint32,
    "uint64_t": np.uint64,

    "int8_t":  np.int8,
    "int16_t": np.int16,
    "int32_t": np.int32,
    "int64_t": np.int64,

    "float":  np.float32,
    "double": np.float64
}
# fmt: on


def make_out_dtypes(enums: list[Enum]) -> dict[str, type]:
    type_map = dict(CTYPE_TO_DTYPE)

    for deferred_type, base_type in DEFERRED_TYPES.items():
        type_map[deferred_type] = CTYPE_TO_DTYPE[base_type]

    for enum in enums:
        type_map[enum.name] = CTYPE_TO_DTYPE[enum.base_type]

    return type_map


def make_enum_encoder(enum: Enum) -> dict[str, int]:
    return {c: i for i, c in enumerate(enum.print_consts)}


@attrs.define
class NodeTableMeta:
    columns: list[str]
    enum_encoders: dict[str, dict[str, int]]
    out_dtypes: dict[str, type]
    dataset_names: dict[str, str]
    key: str

    @classmethod
    def from_source_cpu(cls, source: SourceCPU) -> NodeTableMeta:
        out_dtypes_all = make_out_dtypes(source.enums)
        enum_encoders_all = {
            enum.name: make_enum_encoder(enum) for enum in source.enums
        }

        columns = []
        enum_encoders = {}
        out_dtypes = {}
        dataset_names = {}
        for field in source.node_table.fields:
            if not field.is_static:
                continue
            columns.append(field.print_name)
            if field.type in enum_encoders_all:
                enum_encoders[field.print_name] = enum_encoders_all[field.type]
            out_dtypes[field.print_name] = out_dtypes_all[field.type]
            dataset_names[field.print_name] = field.dataset_name
        key = source.node_table.key

        return cls(columns, enum_encoders, out_dtypes, dataset_names, key)


@attrs.define
class EdgeTableMeta:
    columns: list[str]
    enum_encoders: dict[str, dict[str, int]]
    out_dtypes: dict[str, type]
    dataset_names: dict[str, str]
    target_node_key: str
    source_node_key: str
    edge_index_dtype: type

    @classmethod
    def from_source_cpu(cls, source: SourceCPU) -> EdgeTableMeta:
        out_dtypes_all = make_out_dtypes(source.enums)
        enum_encoders_all = {
            enum.name: make_enum_encoder(enum) for enum in source.enums
        }

        columns = []
        enum_encoders = {}
        out_dtypes = {}
        dataset_names = {}
        for field in source.edge_table.fields:
            if not field.is_static:
                continue
            columns.append(field.print_name)
            if field.type in enum_encoders_all:
                enum_encoders[field.print_name] = enum_encoders_all[field.type]
            out_dtypes[field.print_name] = out_dtypes_all[field.type]
            dataset_names[field.print_name] = field.dataset_name

        target_node_key = source.edge_table.target_node_key
        source_node_key = source.edge_table.source_node_key

        out_dtypes["_target_node_index"] = out_dtypes_all["node_index_type"]
        out_dtypes["_source_node_index"] = out_dtypes_all["node_index_type"]
        edge_index_dtype = out_dtypes_all["edge_index_type"]

        return cls(
            columns,
            enum_encoders,
            out_dtypes,
            dataset_names,
            target_node_key,
            source_node_key,
            edge_index_dtype,
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


def make_source_cpu(simulation_file: Path) -> SourceCPU:
    simulation_bytes = simulation_file.read_bytes()
    pt = mk_pt(str(simulation_file), simulation_bytes)
    ast1 = mk_ast1(simulation_file, pt)
    source = SourceCPU.make(ast1)
    return source


def make_node_table(node_file: Path, ntm: NodeTableMeta) -> pl.DataFrame:
    node_table = read_table(node_file, ntm.columns)
    for col, encoder in ntm.enum_encoders.items():
        node_table = node_table.with_columns(node_table[col].replace(encoder))
    check_null(node_table, "node table")
    return node_table


def make_edge_table(edge_file: Path, etm: EdgeTableMeta, key_idx: dict) -> pl.DataFrame:
    edge_table = read_table(edge_file, etm.columns)
    for col, encoder in etm.enum_encoders.items():
        edge_table = edge_table.with_columns(edge_table[col].replace(encoder))
    check_null(edge_table, "edge table")

    edge_table = edge_table.with_columns(
        edge_table[etm.target_node_key].replace(key_idx).alias("_target_node_index"),
        edge_table[etm.source_node_key].replace(key_idx).alias("_source_node_index"),
    )
    return edge_table


def make_in_inc_csr_indptr(
    edge_table: pl.DataFrame, Vn: int, etm: EdgeTableMeta
) -> np.ndarray:
    target_node_index = edge_table["_target_node_index"].to_numpy()
    edge_count = np.bincount(target_node_index, minlength=Vn)
    cum_edge_count = np.cumsum(edge_count)
    in_inc_csr_indptr = np.zeros(Vn + 1, dtype=etm.edge_index_dtype)
    in_inc_csr_indptr[1:] = cum_edge_count
    return in_inc_csr_indptr


def make_input_file_cpu(
    data_file: Path,
    node_table: pl.DataFrame,
    edge_table: pl.DataFrame,
    ntm: NodeTableMeta,
    etm: EdgeTableMeta,
    in_inc_csr_indptr: np.ndarray,
):
    size_dtype = CTYPE_TO_DTYPE[DEFERRED_TYPES["size_type"]]

    with h5.File(data_file, "w") as fobj:
        for col, dtype in ntm.out_dtypes.items():
            data = node_table[col].to_numpy()
            data = np.asarray(data, dtype)
            fobj.create_dataset(f"/node/{col}", data=data)

        for col, dtype in etm.out_dtypes.items():
            data = edge_table[col].to_numpy()
            data = np.asarray(data, dtype)
            fobj.create_dataset(f"/edge/{col}", data=data)

        fobj.create_dataset(INC_CSR_INDPTR_DATASET_NAME, data=in_inc_csr_indptr)

        fobj.attrs.create("num_nodes", len(node_table), dtype=size_dtype)
        fobj.attrs.create("num_edges", len(edge_table), dtype=size_dtype)


def do_read_nodes_df(data_file: Path, ntm: NodeTableMeta) -> pl.DataFrame:
    df = {}
    with h5.File(data_file, "r") as fobj:
        for col in ntm.columns:
            s = fobj[f"/node/{col}"][...]  # type: ignore
            s = pl.Series(s)
            df[col] = s

    for col, encoder in ntm.enum_encoders.items():
        decoder = {v: k for k, v in encoder.items()}
        df[col] = df[col].replace(decoder)

    return pl.DataFrame(df)


def read_nodes_df(simulation_file: Path, input_file: Path) -> pl.DataFrame:
    source = make_source_cpu(simulation_file)
    ntm = NodeTableMeta.from_source_cpu(source)
    return do_read_nodes_df(input_file, ntm)


def do_read_edges_df(data_file: Path, etm: EdgeTableMeta) -> pl.DataFrame:
    df = {}
    with h5.File(data_file, "r") as fobj:
        for col in etm.columns + ["_target_node_index", "_source_node_index"]:
            s = fobj[f"/edge/{col}"][...]  # type: ignore
            s = pl.Series(s)
            df[col] = s

    for col, encoder in etm.enum_encoders.items():
        decoder = {v: k for k, v in encoder.items()}
        df[col] = [decoder[v] for v in df[col]]

    return pl.DataFrame(df)


def read_edges_df(simulation_file: Path, input_file: Path) -> pl.DataFrame:
    source = make_source_cpu(simulation_file)
    etm = EdgeTableMeta.from_source_cpu(source)
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
    in_inc_csr_indptr = make_in_inc_csr_indptr(edge_table, Vn, etm)

    En = len(edge_table)
    print("### num_edges: ", En)

    # Step 7: Create the data file.
    rich.print("[cyan]Creating input file.[/cyan]")
    make_input_file_cpu(input_file, node_table, edge_table, ntm, etm, in_inc_csr_indptr)

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
        source = make_source_cpu(simulation_file)
        ntm = NodeTableMeta.from_source_cpu(source)
        etm = EdgeTableMeta.from_source_cpu(source)

        do_prepare_input(ntm, etm, node_file, edge_file, input_file)
    except (ParseTreeConstructionError, ASTConstructionError, CodegenError) as e:
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
    except (ParseTreeConstructionError, ASTConstructionError, CodegenError) as e:
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
    except (ParseTreeConstructionError, ASTConstructionError, CodegenError) as e:
        e.rich_print()
        raise SystemExit(1)
