"""Parsl apps."""

import shlex
import shutil
from pathlib import Path
from textwrap import dedent
from typing import Callable, ParamSpec
from concurrent.futures import as_completed


import rich
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    SpinnerColumn,
    MofNCompleteColumn,
    TimeElapsedColumn,
)

import parsl
from parsl.dataflow.futures import AppFuture
from parsl.app.errors import BashExitFailure


def make_fresh_dir(dir: Path) -> None:
    if dir.exists():
        shutil.rmtree(dir)
    dir.mkdir(mode=0o770, parents=True, exist_ok=False)


def cmd_str(input: str) -> str:
    x = dedent(input)
    x = x.strip()
    x = shlex.split(x)
    x = " ".join(x)
    return x


Param = ParamSpec("Param")


def serial_bash_app(func: Callable[Param, str]) -> Callable[Param, AppFuture]:
    bash_app = parsl.bash_app(executors=["serial"])
    return bash_app(func)


def parallel_bash_app(func: Callable[Param, str]) -> Callable[Param, AppFuture]:
    bash_app = parsl.bash_app(executors=["parallel"])
    return bash_app(func)


def wait_for_tasks(message: str, *tasks: AppFuture) -> list:
    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        SpinnerColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    )
    results = []
    with progress:
        p_task = progress.add_task(f"{message} ...", total=len(tasks))
        for task in as_completed(tasks):
            task_tid = task.tid  # type: ignore
            try:
                results.append(task.result())
                progress.advance(p_task)
            except BashExitFailure as e:
                rich.print(
                    f"[red]Task failed: tid={task_tid}, app_name={e.app_name}, exitcode={e.exitcode}[/red]"
                )
                raise SystemExit(1)

    return results


def process_tasks(message: str, tasks: list[AppFuture]) -> list:
    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    )
    results = []
    with progress:
        p_task = progress.add_task(f"{message} ...", total=len(tasks))
        for task in as_completed(tasks):
            task_tid = task.tid  # type: ignore
            try:
                results.append(task.result())
                progress.advance(p_task)
            except BashExitFailure as e:
                rich.print(
                    f"[red]Task failed: tid={task_tid}, app_name={e.app_name}, exitcode={e.exitcode}[/red]"
                )
                raise SystemExit(1)

    return results


def make_file_list(files: list[Path], file_list: Path) -> None:
    with open(file_list, "wt") as fobj:
        for file in files:
            fobj.write(str(file) + "\n")


@parallel_bash_app
def smn_get_dataset_stats(
    mpirun_command: str,
    input_file_list: Path,
    unique_cols: list[str],
    output_file: Path,
) -> str:
    unique_cols = [f"--unique-column '{c}'" for c in unique_cols]
    unique_cols_str = " ".join(unique_cols)

    cmd = f"""
    {mpirun_command} smn get-dataset-stats
        --input-file-list '{input_file_list!s}'
        {unique_cols_str}
        > '{output_file!s}'
    """
    return cmd_str(cmd)


@parallel_bash_app
def smn_shuffle_dataset(
    mpirun_command: str,
    shuffle_column: str,
    input_file_list: Path,
    output_file_list: Path,
) -> str:
    cmd = f"""
    {mpirun_command} smn shuffle-dataset
        --shuffle-column '{shuffle_column}'
        --input-file-list '{input_file_list!s}'
        --output-file-list '{output_file_list!s}'
    """
    return cmd_str(cmd)


def shuffle_dataset(
    mpirun_command: str,
    shuffle_column: str,
    input_files: list[Path],
    output_dir: Path,
    num_output_files: int,
    output_file_template: str,
    unique_cols: list[str],
) -> list[Path]:
    make_fresh_dir(output_dir)

    input_file_list = output_dir / "_input_files.txt"
    make_file_list(input_files, input_file_list)

    output_files = []
    for idx in range(num_output_files):
        output_file = output_dir / output_file_template.format(K=idx)
        output_files.append(output_file)
    output_file_list = output_dir / "_output_files.txt"
    make_file_list(output_files, output_file_list)

    wait_for_tasks(
        "Shuffling dataset",
        smn_shuffle_dataset(
            mpirun_command, shuffle_column, input_file_list, output_file_list
        ),
    )

    wait_for_tasks(
        "Computing dataset stats",
        smn_get_dataset_stats(
            mpirun_command,
            output_file_list,
            unique_cols,
            output_dir / "_dataset_stats.txt",
        ),
    )

    return output_files


@parallel_bash_app
def smn_make_dataset_partitions_approx(
    mpirun_command: str,
    partition_column: str,
    input_file_list: Path,
    num_parts: int,
    partition_file: Path,
) -> str:
    cmd = f"""
    {mpirun_command} smn make-dataset-partitions-approx
        --partition-column '{partition_column}'
        --num-parts {num_parts}
        --input-file-list '{input_file_list!s}'
        --output '{partition_file!s}'
    """
    return cmd_str(cmd)


@serial_bash_app
def smn_make_dataset_partitions_exact(
    partition_column: str,
    input_file_list: Path,
    num_parts: int,
    partition_file: Path,
) -> str:
    cmd = f"""
    smn make-dataset-partitions-exact
        --partition-column '{partition_column}'
        --num-parts {num_parts}
        --input-file-list '{input_file_list!s}'
        --output '{partition_file!s}'
    """
    return cmd_str(cmd)


@parallel_bash_app
def smn_partition_dataset(
    mpirun_command: str,
    partition_column: str,
    input_file_list: Path,
    output_file_list: Path,
    partition_file: Path,
) -> str:
    cmd = f"""
    {mpirun_command} smn partition-dataset
        --partition-column '{partition_column}'
        --input-file-list '{input_file_list!s}'
        --output-file-list '{output_file_list!s}'
        --partition-file '{partition_file!s}'
    """
    return cmd_str(cmd)


def partition_dataset(
    mpirun_command: str,
    exact_partition: bool,
    partition_column: str,
    input_files: list[Path],
    output_dir: Path,
    num_output_files: int,
    output_file_template: str,
    unique_cols: list[str],
) -> list[Path]:
    make_fresh_dir(output_dir)

    input_file_list = output_dir / "_input_files.txt"
    make_file_list(input_files, input_file_list)

    output_files = []
    for idx in range(num_output_files):
        output_file = output_dir / output_file_template.format(K=idx)
        output_files.append(output_file)
    output_file_list = output_dir / "_output_files.txt"
    make_file_list(output_files, output_file_list)

    partition_file = output_dir / "_partition.json"
    if exact_partition:
        wait_for_tasks(
            "Computing exact partitions",
            smn_make_dataset_partitions_exact(
                partition_column, input_file_list, num_output_files, partition_file
            ),
        )
    else:
        wait_for_tasks(
            "Computing approx partitions",
            smn_make_dataset_partitions_approx(
                mpirun_command,
                partition_column,
                input_file_list,
                num_output_files,
                partition_file,
            ),
        )

    wait_for_tasks(
        "Partitioning dataset",
        smn_partition_dataset(
            mpirun_command,
            partition_column,
            input_file_list,
            output_file_list,
            partition_file,
        ),
    )

    wait_for_tasks(
        "Computing dataset stats",
        smn_get_dataset_stats(
            mpirun_command,
            output_file_list,
            unique_cols,
            output_dir / "_dataset_stats.txt",
        ),
    )

    return output_files


@serial_bash_app
def smn_sort_data(
    input_file: Path,
    output_file: Path,
    sort_columns: list[str],
) -> str:
    sort_columns_str = ",".join(sort_columns)
    cmd = f"""
    smn sort-data
        --input '{input_file!s}'
        --output '{output_file!s}'
        --sort-columns {sort_columns_str}
    """
    return cmd_str(cmd)


def sort_files(
    input_files: list[Path],
    output_dir: Path,
    output_file_template: str,
    sort_columns: list[str],
) -> list[Path]:
    make_fresh_dir(output_dir)

    tasks = []
    output_files = []
    for i, input_file in enumerate(input_files):
        output_file = output_dir / output_file_template.format(K=i)
        tasks.append(
            smn_sort_data(
                input_file,
                output_file,
                sort_columns,
            )
        )
        output_files.append(output_file)

    process_tasks("Sorting files", tasks)
    return output_files


@serial_bash_app
def smn_splitcsv(
    input_file: Path,
    num_batches_per_file: int,
    output_dir: Path,
    output_file_template: str,
) -> str:
    cmd = f"""
    smn splitcsv
        --input-file '{input_file!s}'
        --num-batches-per-file {num_batches_per_file}
        --output-dir '{output_dir!s}'
        --output-file-template {output_file_template}
    """
    return cmd_str(cmd)


def splitcsv_files(
    input_files: list[Path],
    num_batches_per_file: int,
    output_dir: Path,
    output_prefix: str,
) -> list[Path]:
    make_fresh_dir(output_dir)

    tasks = []
    for i, input_file in enumerate(input_files):
        output_file_template = f"{output_prefix}-{i}-{{K}}.csv"
        tasks.append(
            smn_splitcsv(
                input_file, num_batches_per_file, output_dir, output_file_template
            )
        )

    process_tasks("Splitting csv files", tasks)

    gen_glob = f"{output_prefix}-*-*.csv"
    fnames = output_dir.glob(gen_glob)
    fnames = list(fnames)

    return fnames


@serial_bash_app
def smn_make_lid_ltype_map(
    input_file: Path,
    output_file: Path,
) -> str:
    cmd = f"""
    smn preprocess-activity-file make-lid-ltype-map
        --input '{input_file!s}'
        --output '{output_file!s}'
    """
    return cmd_str(cmd)


@serial_bash_app
def smn_make_pid_hid_map(
    input_file: Path,
    output_file: Path,
) -> str:
    cmd = f"""
    smn preprocess-activity-file make-pid-hid-map
        --input '{input_file!s}'
        --output '{output_file!s}'
    """
    return cmd_str(cmd)


@serial_bash_app
def smn_preprocess_activity_file(
    input_file: Path,
    output_file: Path,
    min_start_time: int,
    max_start_time: int,
    pid_hid_map_file: Path,
    lid_ltype_map_file: Path,
) -> str:
    cmd = f"""
    smn preprocess-activity-file preprocess-file
        --input '{input_file!s}'
        --output '{output_file!s}'
        --pid-hid-map-file '{pid_hid_map_file!s}'
        --lid-ltype-map-file '{lid_ltype_map_file!s}'
        --min-start-time {min_start_time}
        --max-start-time {max_start_time}
    """
    return cmd_str(cmd)


def preprocess_activity_files(
    location_file: Path,
    person_file: Path,
    input_files: list[Path],
    output_dir: Path,
    min_start_time: int,
    max_start_time: int,
) -> list[Path]:
    make_fresh_dir(output_dir)

    lid_ltype_map_file = output_dir / "lid_ltype_map.pickle"
    pid_hid_map_file = output_dir / "pid_hid_map_file.pickle"

    wait_for_tasks(
        "Making lid -> ltype and pid -> hid maps",
        smn_make_lid_ltype_map(location_file, lid_ltype_map_file),
        smn_make_pid_hid_map(person_file, pid_hid_map_file),
    )

    output_files = []
    tasks = []
    for i, input_file in enumerate(input_files):
        output_file = output_dir / f"activity-{i}.parquet"
        output_files.append(output_file)
        tasks.append(
            smn_preprocess_activity_file(
                input_file,
                output_file,
                min_start_time,
                max_start_time,
                pid_hid_map_file,
                lid_ltype_map_file,
            )
        )

    process_tasks("Preprocessing activity files", tasks)
    return output_files


@parallel_bash_app
def smn_make_lid_stats(
    mpirun_command: str,
    input_file_list: Path,
    output_file_list: Path,
    min_expected_degree: int,
    max_expected_degree: int,
    alpha: float,
    measure: str,
) -> str:
    cmd = f"""
    {mpirun_command} smn make-lid-stats
        --input-file-list '{input_file_list!s}'
        --output-file-list '{output_file_list!s}'
        --min-expected-degree {min_expected_degree}
        --max-expected-degree {max_expected_degree}
        --alpha {alpha}
        --measure {measure}
    """
    return cmd_str(cmd)


@parallel_bash_app
def smn_get_edge_count_dist(
    mpirun_command: str, stats_file_list: Path, output_file: Path
) -> str:
    cmd = f"""
    {mpirun_command} smn get-edge-count-dist
        --stats-file-list '{stats_file_list!s}'
        > '{output_file!s}'
    """
    return cmd_str(cmd)


def make_lid_stats(
    mpirun_command: str,
    input_files: list[Path],
    output_dir: Path,
    min_expected_degree: int,
    max_expected_degree: int,
    alpha: float,
    measure: str,
) -> list[Path]:
    make_fresh_dir(output_dir)

    input_file_list = output_dir / "_input_files.txt"
    make_file_list(input_files, input_file_list)

    num_work_files = len(input_files)

    output_files = []
    for i in range(num_work_files):
        output_file = output_dir / f"location-stats-{i}.parquet"
        output_files.append(output_file)

    output_file_list = output_dir / "_output_files.txt"
    make_file_list(output_files, output_file_list)

    wait_for_tasks(
        "Computing LID stats",
        smn_make_lid_stats(
            mpirun_command,
            input_file_list,
            output_file_list,
            min_expected_degree,
            max_expected_degree,
            alpha,
            measure,
        ),
    )

    wait_for_tasks(
        "Computing dataset stats for LID stats files",
        smn_get_dataset_stats(
            mpirun_command,
            output_file_list,
            ["lid"],
            output_dir / "_dataset_stats.txt",
        ),
        smn_get_edge_count_dist(
            mpirun_command,
            output_file_list,
            output_dir / "_edge_count_dist.txt",
        ),
    )

    return output_files


@parallel_bash_app
def smn_create_network(
    mpirun_command: str,
    activity_file_list: Path,
    stats_file_list: Path,
    contact_file_list: Path,
) -> str:
    cmd = f"""
    {mpirun_command} smn create-network
        --activity-file-list '{activity_file_list!s}'
        --stats-file-list '{stats_file_list!s}'
        --contact-file-list '{contact_file_list!s}'
    """
    return cmd_str(cmd)


def create_network(
    mpirun_command: str,
    activity_files: list[Path],
    lid_stats_files: list[Path],
    output_dir: Path,
) -> AppFuture:
    make_fresh_dir(output_dir)

    num_work_files = len(activity_files)

    activity_file_list = output_dir / "_activity_files.txt"
    make_file_list(activity_files, activity_file_list)

    stats_file_list = output_dir / "_stats_files.txt"
    make_file_list(lid_stats_files, stats_file_list)

    contact_files = [
        output_dir / f"contacts-{i}.parquet" for i in range(num_work_files)
    ]
    contact_file_list = output_dir / "_output_files.txt"
    make_file_list(contact_files, contact_file_list)

    return smn_create_network(
        mpirun_command,
        activity_file_list,
        stats_file_list,
        contact_file_list,
    )


def make_network_stats(mpirun_command: str, network_dir: Path) -> AppFuture:
    contact_file_list = network_dir / "_output_files.txt"
    dataset_stats_file = network_dir / "_dataset_stats.txt"

    return smn_get_dataset_stats(
        mpirun_command,
        contact_file_list,
        ["pid1", "pid2", "lid"],
        dataset_stats_file,
    )
