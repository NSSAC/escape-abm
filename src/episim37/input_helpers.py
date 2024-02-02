"""Helper utilities for creating/reading input.h5 file."""

from __future__ import annotations

from pathlib import Path

import attrs
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
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


def make_in_dtypes(enums: list[Enum]) -> dict[str, type]:
    type_map = dict(CTYPE_TO_DTYPE)

    for deferred_type, base_type in DEFERRED_TYPES.items():
        type_map[deferred_type] = CTYPE_TO_DTYPE[base_type]

    for enum in enums:
        type_map[enum.name] = str

    return type_map


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
    in_dtypes: dict[str, type]
    enum_encoders: dict[str, dict[str, int]]
    out_dtypes: dict[str, type]
    dataset_names: dict[str, str]
    key: str

    @classmethod
    def from_source_cpu(cls, source: SourceCPU) -> NodeTableMeta:
        in_dtypes_all = make_in_dtypes(source.enums)
        out_dtypes_all = make_out_dtypes(source.enums)
        enum_encoders_all = {
            enum.name: make_enum_encoder(enum) for enum in source.enums
        }

        columns = []
        in_dtypes = {}
        enum_encoders = {}
        out_dtypes = {}
        dataset_names = {}
        for field in source.node_table.fields:
            if not field.is_static:
                continue
            columns.append(field.print_name)
            in_dtypes[field.print_name] = in_dtypes_all[field.type]
            if field.type in enum_encoders_all:
                enum_encoders[field.print_name] = enum_encoders_all[field.type]
            out_dtypes[field.print_name] = out_dtypes_all[field.type]
            dataset_names[field.print_name] = field.dataset_name
        key = source.node_table.key

        return cls(columns, in_dtypes, enum_encoders, out_dtypes, dataset_names, key)


@attrs.define
class EdgeTableMeta:
    columns: list[str]
    in_dtypes: dict[str, type]
    enum_encoders: dict[str, dict[str, int]]
    out_dtypes: dict[str, type]
    dataset_names: dict[str, str]
    target_node_key: str
    source_node_key: str
    edge_index_dtype: type

    @classmethod
    def from_source_cpu(cls, source: SourceCPU) -> EdgeTableMeta:
        in_dtypes_all = make_in_dtypes(source.enums)
        out_dtypes_all = make_out_dtypes(source.enums)
        enum_encoders_all = {
            enum.name: make_enum_encoder(enum) for enum in source.enums
        }

        columns = []
        in_dtypes = {}
        enum_encoders = {}
        out_dtypes = {}
        dataset_names = {}
        for field in source.edge_table.fields:
            if not field.is_static:
                continue
            columns.append(field.print_name)
            in_dtypes[field.print_name] = in_dtypes_all[field.type]
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
            in_dtypes,
            enum_encoders,
            out_dtypes,
            dataset_names,
            target_node_key,
            source_node_key,
            edge_index_dtype,
        )


def read_table(file: Path, columns: list, dtype: dict) -> pd.DataFrame:
    if file.suffix == ".csv":
        return pd.read_csv(file, usecols=columns, dtype=dtype)  # type: ignore
    elif file.suffix == ".parquet":
        schema = pa.schema([(c, pa.from_numpy_dtype(t)) for c, t in dtype.items()])
        return pq.read_table(file, columns=columns, schema=schema).to_pandas()
    else:
        rich.print("[red]Unknown filetype for node file[/red]")
        raise SystemExit(1)


def check_null(table: pd.DataFrame, name: str):
    for col in table.columns:
        if table[col].isnull().any():  # type: ignore
            rich.print(f"[red]Column {col} in {name} has null values.[/red]")
            raise SystemExit(1)


def make_source_cpu(simulation_file: Path) -> SourceCPU:
    simulation_bytes = simulation_file.read_bytes()
    try:
        pt = mk_pt(str(simulation_file), simulation_bytes)
        ast1 = mk_ast1(simulation_file, pt)
        source = SourceCPU.make(ast1)
        return source
    except (ParseTreeConstructionError, ASTConstructionError, CodegenError) as e:
        e.rich_print()
        raise SystemExit(1)


def make_node_table(node_file: Path, ntm: NodeTableMeta) -> pd.DataFrame:
    node_table = read_table(node_file, ntm.columns, ntm.in_dtypes)
    for col, encoder in ntm.enum_encoders.items():
        node_table[col] = node_table[col].map(encoder)  # type: ignore
    check_null(node_table, "node table")
    return node_table


def make_edge_table(edge_file: Path, etm: EdgeTableMeta, key_idx: dict) -> pd.DataFrame:
    edge_table = read_table(edge_file, etm.columns, etm.in_dtypes)
    for col, encoder in etm.enum_encoders.items():
        edge_table[col] = edge_table[col].map(encoder)  # type: ignore
    check_null(edge_table, "edge table")

    edge_table["_target_node_index"] = edge_table[etm.target_node_key].map(key_idx)  # type: ignore
    edge_table["_source_node_index"] = edge_table[etm.source_node_key].map(key_idx)  # type: ignore
    return edge_table


def make_in_inc_csr_indptr(
    edge_table: pd.DataFrame, Vn: int, etm: EdgeTableMeta
) -> np.ndarray:
    target_node_index = edge_table._target_node_index.to_numpy()
    edge_count = np.bincount(target_node_index, minlength=Vn)
    cum_edge_count = np.cumsum(edge_count)
    in_inc_csr_indptr = np.zeros(Vn + 1, dtype=etm.edge_index_dtype)
    in_inc_csr_indptr[1:] = cum_edge_count
    return in_inc_csr_indptr


def make_input_file_cpu(
    data_file: Path,
    node_table: pd.DataFrame,
    edge_table: pd.DataFrame,
    ntm: NodeTableMeta,
    etm: EdgeTableMeta,
    in_inc_csr_indptr: np.ndarray,
):
    size_dtype = CTYPE_TO_DTYPE[DEFERRED_TYPES["size_type"]]

    with h5.File(data_file, "w") as fobj:
        for col, dtype in ntm.out_dtypes.items():
            fobj.create_dataset(f"/node/{col}", data=node_table[col].to_numpy(dtype))

        for col, dtype in etm.out_dtypes.items():
            fobj.create_dataset(f"/edge/{col}", data=edge_table[col].to_numpy(dtype))

        fobj.create_dataset(INC_CSR_INDPTR_DATASET_NAME, data=in_inc_csr_indptr)

        fobj.attrs.create("num_nodes", len(node_table), dtype=size_dtype)
        fobj.attrs.create("num_edges", len(edge_table), dtype=size_dtype)


@click.command()
@click.option(
    "-s",
    "--simulation",
    "simulation_file",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to simulation (esl37) file.",
)
@click.option(
    "-n",
    "--node",
    "node_file",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to node attributes file.",
)
@click.option(
    "-e",
    "--edge",
    "edge_file",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to edge attributes file.",
)
@click.option(
    "-d",
    "--data",
    "data_file",
    type=click.Path(exists=False, file_okay=True, dir_okay=False, path_type=Path),
    default="output.h5",
    show_default=True,
    help="Path to data file for simulator.",
)
def prepare_input(
    simulation_file: Path, node_file: Path, edge_file: Path, data_file: Path
):
    """Prepare input for simulation."""
    rich.print("[cyan]Parsing simulator code.[/cyan]")
    source = make_source_cpu(simulation_file)
    ntm = NodeTableMeta.from_source_cpu(source)
    etm = EdgeTableMeta.from_source_cpu(source)

    # Step 1: Preprocess the node table files
    rich.print("[cyan]Reading node table.[/cyan]")
    node_table = make_node_table(node_file, ntm)

    # Step 2: Sort the node table dataset
    rich.print("[cyan]Sorting node table.[/cyan]")
    node_table.sort_values(ntm.key, inplace=True, ignore_index=True)

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
    edge_table.sort_values(
        ["_target_node_index", "_source_node_index"], inplace=True, ignore_index=True
    )

    # Step 6: Make incoming incidence CSR graph's indptr
    rich.print("[cyan]Computing incoming incidence CSR graph's indptr.[/cyan]")
    in_inc_csr_indptr = make_in_inc_csr_indptr(edge_table, Vn, etm)

    En = len(edge_table)
    print("### num_edges: ", En)

    # Step 7: Create the data file.
    rich.print("[cyan]Creating data file.[/cyan]")
    make_input_file_cpu(data_file, node_table, edge_table, ntm, etm, in_inc_csr_indptr)

    rich.print("[green]Data file created successfully.[/green]")
