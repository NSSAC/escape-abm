"""Prepare input data."""

from pathlib import Path

import numpy as np
import scipy.sparse as sparse
import h5py as h5
import pyarrow as pa
import pyarrow.parquet as pq

import click
import rich

from .ast1 import mk_ast1, ASTConstructionError
from .parse_tree import mk_pt, ParseTreeConstructionError
from .codegen_cpu import (
    CodegenError,
    INC_CSR_INDPTR_DATASET_NAME,
    EnumType as Enum,
    Source as SourceCPU,
    DEFERRED_TYPES as DEFERRED_TYPES_CPU,
)

# fmt: off
CTYPE_TO_DTYPE = {
    "uint8_t": np.uint8,
    "uint16_t": np.uint16,
    "uint32_t": np.uint32,
    "uint64_t": np.uint64,

    "int8_t": np.int8,
    "int16_t": np.int16,
    "int32_t": np.int32,
    "int64_t": np.int64,

    "float": np.float32,
    "double": np.float64
}
# fmt: on


def type_to_in_dtype(type: str, deferred_types: dict[str, str], enums: list[Enum]):
    for enum in enums:
        if type == enum.name:
            return str

    if type not in CTYPE_TO_DTYPE:
        if type in deferred_types:
            type = deferred_types[type]

    if type not in CTYPE_TO_DTYPE:
        raise ValueError(f"Unknown type {type}")

    return CTYPE_TO_DTYPE[type]


def type_to_out_dtype(type: str, deferred_types: dict[str, str], enums: list[Enum]):
    for enum in enums:
        if type == enum.name:
            type = enum.base_type
            break

    if type not in CTYPE_TO_DTYPE:
        if type in deferred_types:
            type = deferred_types[type]

    if type not in CTYPE_TO_DTYPE:
        raise ValueError(f"Unknown type {type}")

    return CTYPE_TO_DTYPE[type]


def enum_converter(type: str, enums: list[Enum]):
    for enum in enums:
        if type == enum.name:
            return {c: i for i, c in enumerate(enum.print_consts)}

    return None


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
    help="Path to node attributes (parquet) file.",
)
@click.option(
    "-e",
    "--edge",
    "edge_file",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to edge attributes (parquet) file.",
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
def prepare_cpu(
    simulation_file: Path, node_file: Path, edge_file: Path, data_file: Path
):
    """Prepare input for simulation on CPUs."""
    simulation_bytes = simulation_file.read_bytes()

    rich.print("[yellow]Parsing simulator code.[/yellow]")
    try:
        pt = mk_pt(str(simulation_file), simulation_bytes)
        ast1 = mk_ast1(simulation_file, pt)
        source = SourceCPU.make(ast1)
    except (ParseTreeConstructionError, ASTConstructionError, CodegenError) as e:
        e.rich_print()
        return

    rich.print("[yellow]Reading node table.[/yellow]")
    node_columns = []
    node_in_dtypes = {}
    node_enum_converter = {}
    node_out_dtypes = {}
    node_dataset_names = {}
    for field in source.node_table.fields:
        if not field.is_static:
            continue
        node_columns.append(field.print_name)
        node_in_dtypes[field.print_name] = type_to_in_dtype(
            field.type, DEFERRED_TYPES_CPU, source.enums
        )
        converter = enum_converter(field.type, source.enums)
        if converter is not None:
            node_enum_converter[field.print_name] = converter
        node_out_dtypes[field.print_name] = type_to_out_dtype(
            field.type, DEFERRED_TYPES_CPU, source.enums
        )
        node_dataset_names[field.print_name] = field.dataset_name
    node_key = ast1.node_table.key.name
    node_schema = pa.schema(
        [(c, pa.from_numpy_dtype(dtype)) for c, dtype in node_in_dtypes.items()]
    )
    node_table = pq.read_table(
        node_file, columns=node_columns, schema=node_schema
    ).to_pandas()
    for col, converter in node_enum_converter.items():
        node_table[col] = node_table[col].map(converter)

    for col in node_table.columns:
        if node_table[col].isnull().any():
            rich.print(f"[red]Column {col} Node table has null values.[/red]")
            return

    rich.print("[yellow]Reading edge table.[/yellow]")
    edge_columns = []
    edge_in_dtypes = {}
    edge_enum_converter = {}
    edge_out_dtypes = {}
    edge_dataset_names = {}
    for field in source.edge_table.fields:
        if not field.is_static:
            continue
        edge_columns.append(field.print_name)
        edge_in_dtypes[field.print_name] = type_to_in_dtype(
            field.type, DEFERRED_TYPES_CPU, source.enums
        )
        converter = enum_converter(field.type, source.enums)
        if converter is not None:
            edge_enum_converter[field.print_name] = converter
        edge_out_dtypes[field.print_name] = type_to_out_dtype(
            field.type, DEFERRED_TYPES_CPU, source.enums
        )
        edge_dataset_names[field.print_name] = field.dataset_name
    target_node_key = ast1.edge_table.target_node_key.name
    source_node_key = ast1.edge_table.source_node_key.name
    edge_schema = pa.schema(
        [(c, pa.from_numpy_dtype(dtype)) for c, dtype in edge_in_dtypes.items()]
    )
    edge_table = pq.read_table(
        edge_file, columns=edge_columns, schema=edge_schema
    ).to_pandas()
    for col, converter in edge_enum_converter.items():
        edge_table[col] = edge_table[col].map(converter)

    for col in edge_table.columns:
        if edge_table[col].isnull().any():
            rich.print(f"[red]Column {col} in edge table has null values.[/red]")
            return

    Vn = len(node_table)
    En = len(edge_table)
    print("### num_nodes: ", Vn)
    print("### num_edges: ", En)

    node_index_dtype = CTYPE_TO_DTYPE[DEFERRED_TYPES_CPU["node_index_type"]]
    edge_index_dtype = CTYPE_TO_DTYPE[DEFERRED_TYPES_CPU["edge_index_type"]]
    size_dtype = CTYPE_TO_DTYPE[DEFERRED_TYPES_CPU["size_type"]]

    key_idx = {k: idx for idx, k in enumerate(node_table[node_key])}

    rich.print("[yellow]Sorting edges.[/yellow]")
    edge_table["_target_node_index"] = edge_table[target_node_key].map(key_idx)
    edge_table["_source_node_index"] = edge_table[source_node_key].map(key_idx)
    edge_table.sort_values(
        ["_target_node_index", "_source_node_index"], inplace=True, ignore_index=True
    )
    edge_out_dtypes["_target_node_index"] = node_index_dtype
    edge_out_dtypes["_source_node_index"] = edge_index_dtype

    rich.print("[yellow]Computing incoming incidence sparse matrix.[/yellow]")
    I = edge_table._target_node_index.to_numpy()
    J = np.arange(En, dtype=int)
    data = np.ones_like(I)
    in_inc_csr = sparse.coo_matrix((data, (I, J)), shape=(Vn, En))
    in_inc_csr = in_inc_csr.tocsr()

    rich.print("[yellow]Creating data file.[/yellow]")
    with h5.File(data_file, "w") as fobj:
        for col, dtype in node_out_dtypes.items():
            print(f"Creating: /node/{col}")
            fobj.create_dataset(f"/node/{col}", data=node_table[col].to_numpy(dtype))

        for col, dtype in edge_out_dtypes.items():
            print(f"Creating: /edge/{col}")
            fobj.create_dataset(f"/edge/{col}", data=edge_table[col].to_numpy(dtype))

        print(f"Creating: {INC_CSR_INDPTR_DATASET_NAME}")
        fobj.create_dataset(
            INC_CSR_INDPTR_DATASET_NAME,
            data=np.array(in_inc_csr.indptr, dtype=edge_index_dtype),
        )

        fobj.attrs.create("num_nodes", Vn, dtype=size_dtype)
        fobj.attrs.create("num_edges", En, dtype=size_dtype)

    rich.print("[green]Data file created successfully.[/green]")
