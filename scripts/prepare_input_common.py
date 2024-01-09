"""Prepare large input data using Parsl and SMN tool."""

import pickle
from pathlib import Path
from typing import Callable, ParamSpec, Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import rich
import parsl
from parsl.dataflow.futures import AppFuture

from episim37.prepare_input import (
    NodeTableMeta,
    EdgeTableMeta,
    make_source_cpu,
    make_node_table,
    make_edge_table,
    make_input_file_cpu,
)

from parsl_helpers import (
    make_fresh_dir,
    wait_for_tasks,
    process_tasks,
    partition_dataset,
    sort_files,
)

Param = ParamSpec("Param")


def serial_python_app(func: Callable[Param, Any]) -> Callable[Param, AppFuture]:
    python_app = parsl.python_app(executors=["serial"])
    return python_app(func)


@serial_python_app
def preprocess_node_table_file(
    input_file: Path, output_file: Path, ntm: NodeTableMeta
) -> None:
    node_table = make_node_table(input_file, ntm)
    node_table.to_parquet(output_file, index=False)


def preprocess_node_table(
    input_files: list[Path], output_dir: Path, ntm: NodeTableMeta
) -> list[Path]:
    make_fresh_dir(output_dir)

    tasks = []
    output_files = []
    for i, input_file in enumerate(input_files):
        output_file = output_dir / f"node-{i}.parquet"
        tasks.append(preprocess_node_table_file(input_file, output_file, ntm))
        output_files.append(output_file)

    process_tasks("Preprocessing node table", tasks)

    return output_files


@serial_python_app
def create_node_index_file(files: list[Path], key: str, index_file: Path) -> int:
    all_nodes = []
    for file in files:
        nodes = pq.read_table(file, columns=[key])
        nodes = nodes.column(key)
        nodes = nodes.to_numpy()
        all_nodes.append(nodes)
    all_nodes = np.concatenate(all_nodes)

    key_idx = {k.item(): i for i, k in enumerate(all_nodes)}

    with open(index_file, "wb") as fobj:
        pickle.dump(key_idx, fobj, protocol=pickle.HIGHEST_PROTOCOL)
    return len(key_idx)


@serial_python_app
def preprocess_edge_table_file(
    input_file: Path, output_file: Path, etm: EdgeTableMeta, index_file: Path
) -> None:
    with open(index_file, "rb") as fobj:
        key_idx = pickle.load(fobj)

    edge_table = make_edge_table(input_file, etm, key_idx)
    edge_table.to_parquet(output_file, index=False)


def preprocess_edge_table(
    input_files: list[Path], output_dir: Path, etm: EdgeTableMeta, index_file: Path
) -> list[Path]:
    make_fresh_dir(output_dir)

    tasks = []
    output_files = []
    for i, input_file in enumerate(input_files):
        output_file = output_dir / f"edge-{i}.parquet"
        tasks.append(
            preprocess_edge_table_file(input_file, output_file, etm, index_file)
        )
        output_files.append(output_file)

    process_tasks("Preprocessing edge table", tasks)

    return output_files


@serial_python_app
def create_in_inc_csr_indptr_file(
    files: list[Path], Vn: int, etm: EdgeTableMeta, indptr_file: Path
) -> int:
    target_node_index = []
    for file in files:
        ix = pq.read_table(file, columns=["_target_node_index"])
        ix = ix.column("_target_node_index")
        ix = ix.to_numpy()
        target_node_index.append(ix)
    target_node_index = np.concatenate(target_node_index)

    edge_count = np.bincount(target_node_index, minlength=Vn)
    cum_edge_count = np.cumsum(edge_count)
    in_inc_csr_indptr = np.zeros(Vn + 1, dtype=etm.edge_index_dtype)
    in_inc_csr_indptr[1:] = cum_edge_count

    with open(indptr_file, "wb") as fobj:
        pickle.dump(in_inc_csr_indptr, fobj, protocol=pickle.HIGHEST_PROTOCOL)

    return len(target_node_index)


@serial_python_app
def create_input_file_cpu(
    data_file: Path,
    node_files: list[Path],
    edge_files: list[Path],
    ntm: NodeTableMeta,
    etm: EdgeTableMeta,
    indptr_file: Path,
):
    node_table = [pq.read_table(file) for file in node_files]
    node_table = pa.concat_tables(node_table)
    node_table = node_table.to_pandas()

    edge_table = [pq.read_table(file) for file in edge_files]
    edge_table = pa.concat_tables(edge_table)
    edge_table = edge_table.to_pandas()

    with open(indptr_file, "rb") as fobj:
        in_inc_csr_indptr = pickle.load(fobj)

    make_input_file_cpu(data_file, node_table, edge_table, ntm, etm, in_inc_csr_indptr)


def do_prepare_input_cpu(
    mpirun_command: str,
    num_work_files: int,
    input_root: Path,
    output_root: Path,
    simulation_file: Path,
    node_files_glob: str,
    edge_files_glob: str,
    data_file: Path,
):
    rich.print("[cyan]Parsing simulator code.[/cyan]")
    source = make_source_cpu(simulation_file)
    ntm = NodeTableMeta.from_source_cpu(source)
    etm = EdgeTableMeta.from_source_cpu(source)

    node_files = list(input_root.glob(node_files_glob))
    rich.print(f"[cyan]Got {len(node_files)} node files.[/cyan]")

    edge_files = list(input_root.glob(edge_files_glob))
    rich.print(f"[cyan]Got {len(edge_files)} edge files.[/cyan]")

    # Step 1: Preprocess the node table files
    #   * Select only the columns necessary
    #   * Encode the enum columns
    rich.print("[cyan]Reading node table.[/cyan]")
    preproc_node_dir = output_root / "nodes_preproc"
    preproc_node_files = preprocess_node_table(node_files, preproc_node_dir, ntm)

    # Step 2: Sort the node table dataset
    #   * Use exact partition based on node_key
    #   * Sort each file based on node key
    rich.print("[cyan]Sorting node table.[/cyan]")
    node_partition_dir = output_root / "nodes_partition"
    node_partition_files = partition_dataset(
        mpirun_command=mpirun_command,
        exact_partition=True,
        partition_column=ntm.key,
        input_files=preproc_node_files,
        output_dir=node_partition_dir,
        num_output_files=num_work_files,
        output_file_template="node-{K}.parquet",
        unique_cols=[ntm.key],
    )

    node_sort_dir = output_root / "nodes_sorted"
    node_sorted_files = sort_files(
        input_files=node_partition_files,
        output_dir=node_sort_dir,
        output_file_template="node-{K}.parquet",
        sort_columns=[ntm.key],
    )

    # Step 3: Create a node index
    rich.print("[cyan]Making node index.[/cyan]")
    node_index_file = output_root / "node_index.pickle"
    Vn = wait_for_tasks(
        "Creating node index file",
        create_node_index_file(node_sorted_files, ntm.key, node_index_file),
    )[0]
    print("### num_nodes: ", Vn)

    # Step 4: Preprocess the edge table files
    #   * Select only the necessary columns
    #   * Encode the enum columns
    #   * Add source + target node index columns
    rich.print("[cyan]Reading edge table.[/cyan]")
    preproc_edge_dir = output_root / "edges_preproc"
    preproc_edge_files = preprocess_edge_table(
        edge_files, preproc_edge_dir, etm, node_index_file
    )

    # Step 5: Sort the edge table dataset
    #   * Use exact partition by target_node_key
    #   * Sort each file based on target_node_key, source_node_key
    rich.print("[cyan]Sorting edges table.[/cyan]")
    edge_partition_dir = output_root / "edges_partition"
    edge_partition_files = partition_dataset(
        mpirun_command=mpirun_command,
        exact_partition=True,
        partition_column="_target_node_index",
        input_files=preproc_edge_files,
        output_dir=edge_partition_dir,
        num_output_files=num_work_files,
        output_file_template="edge-{K}.parquet",
        unique_cols=[etm.target_node_key, etm.source_node_key],
    )

    edge_sort_dir = output_root / "edges_sorted"
    edge_sorted_files = sort_files(
        input_files=edge_partition_files,
        output_dir=edge_sort_dir,
        output_file_template="edge-{K}.parquet",
        sort_columns=["_target_node_index", "_source_node_index"],
    )

    # Step 6: Make incoming incidence CSR graph's indptr
    rich.print("[cyan]Computing incoming incidence CSR graph's indptr.[/cyan]")
    indptr_file = output_root / "in_inc_csr_indptr.pickle"
    En = wait_for_tasks(
        "Creating incoming incidence CSR",
        create_in_inc_csr_indptr_file(edge_sorted_files, Vn, etm, indptr_file),
    )[0]
    print("### num_edges: ", En)

    # Step 7: Create the data file
    rich.print("[cyan]Creating data file.[/cyan]")
    wait_for_tasks(
        "Creating input file.",
        create_input_file_cpu(
            data_file, node_sorted_files, edge_sorted_files, ntm, etm, indptr_file
        ),
    )

    rich.print("[green]Data file created successfully.[/green]")
