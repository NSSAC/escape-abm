"""EpiSim37 utilities."""

from pathlib import Path
from typing import Any

import h5py as h5
import polars as pl

from .input_helpers import (
    do_read_edges_df,
    do_read_nodes_df,
)
from .output_helpers import (
    do_extract_summary,
    do_extract_transitions,
    do_extract_interventions,
    do_extract_transmissions,
    find_contagion,
)
from .parse_tree import mk_pt
from .ast1 import mk_ast1
from .codegen_cpu import (
    mk_ir as mk_cpu_ir,
    do_prepare as do_prepare_cpu,
    do_build as do_build_cpu,
    do_simulate as do_simulate_cpu,
)
from .input_helpers import (
    NodeTableMeta,
    EdgeTableMeta,
    make_node_table,
    make_edge_table,
    make_in_inc_csr_indptr,
    make_input_file_cpu,
)


class CPUSimulator:
    """Simulate on CPU."""

    def __init__(
        self,
        simulation_file: str | Path,
        node_file: str | Path,
        edge_file: str | Path,
        work_dir: str | Path,
    ):
        self.simulation_file = Path(simulation_file)
        self.node_file = Path(node_file)
        self.edge_file = Path(edge_file)
        self.work_dir = Path(work_dir)

        assert self.simulation_file.exists(), "Simulation file doesn't exist"
        assert self.node_file.exists(), "Node file doesn't exist"
        assert self.edge_file.exists(), "Edge file doesn't exist"

        self.pt = mk_pt(str(self.simulation_file), self.simulation_file.read_bytes())
        self.ast1 = mk_ast1(self.simulation_file, self.pt)
        self.ir = mk_cpu_ir(self.ast1)

        self.ntm = NodeTableMeta.from_source_cpu(self.ir)
        self.etm = EdgeTableMeta.from_source_cpu(self.ir)

        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.input_file = self.work_dir / "input.h5"
        self.gen_code_dir = self.work_dir / "gen_code"
        self.gen_code_dir.mkdir(parents=True, exist_ok=True)

        self.output_files: list[Path] = []

    def prepare_input(self) -> None:
        ntm = NodeTableMeta.from_source_cpu(self.ir)
        etm = EdgeTableMeta.from_source_cpu(self.ir)

        # Step 1: Preprocess the node table files
        print("Reading node table.")
        node_table = make_node_table(self.node_file, ntm)

        # Step 2: Sort the node table dataset
        print("Sorting node table")
        node_table = node_table.sort(ntm.key)

        # Step 3: Create a node index
        print("Making node index.")
        key_idx = {k: idx for idx, k in enumerate(node_table[ntm.key])}

        Vn = len(node_table)
        print("### num_nodes: ", Vn)

        # Step 4: Preprocess the edge table files
        print("Reading edge table.")
        edge_table = make_edge_table(self.edge_file, etm, key_idx)

        # Step 5: Sort the edge table dataset
        print("Sorting edge table.")
        edge_table = edge_table.sort(["_target_node_index", "_source_node_index"])

        # Step 6: Make incoming incidence CSR graph's indptr
        print("Computing incoming incidence CSR graph's indptr.")
        in_inc_csr_indptr = make_in_inc_csr_indptr(edge_table, Vn, etm)

        En = len(edge_table)
        print("### num_edges: ", En)

        # Step 7: Create the data file.
        print("Creating input file.")
        make_input_file_cpu(
            self.input_file, node_table, edge_table, ntm, etm, in_inc_csr_indptr
        )

        print("Input file created successfully.")

    def prepare_build(self) -> None:
        do_prepare_cpu(self.gen_code_dir, self.simulation_file, self.ir)

    def build(self) -> None:
        do_build_cpu(self.gen_code_dir)

    def simulate(self, num_ticks: int = 0, configs: dict[str, Any] = {}) -> Path:
        i = len(self.output_files)
        output_file = self.gen_code_dir / f"output-{i}.h5"
        do_simulate_cpu(
            self.gen_code_dir,
            input_file=self.input_file,
            output_file=output_file,
            num_ticks=num_ticks,
            configs=configs,
        )
        self.output_files.append(output_file)
        return output_file

    def extract_nodes(self) -> pl.DataFrame:
        assert self.input_file.exists(), "Input file doesn't exist."
        return do_read_nodes_df(self.input_file, self.ntm)

    def extract_edges(self) -> pl.DataFrame:
        assert self.input_file.exists(), "Input file doesn't exist."
        return do_read_edges_df(self.input_file, self.etm)

    def extract_summary(
        self, output_file: Path, contagion_name: str = ""
    ) -> pl.DataFrame:
        contagion = find_contagion(contagion_name, self.ast1)
        with h5.File(output_file, "r") as sim_output:
            df = do_extract_summary(sim_output, contagion)
        return df

    def extract_transitions(
        self, output_file: Path, contagion_name: str = ""
    ) -> pl.DataFrame:
        contagion = find_contagion(contagion_name, self.ast1)
        with h5.File(output_file, "r") as sim_output:
            df = do_extract_transitions(sim_output, contagion)
        return df

    def extract_interventions(
        self, output_file: Path, contagion_name: str = ""
    ) -> pl.DataFrame:
        contagion = find_contagion(contagion_name, self.ast1)
        with h5.File(output_file, "r") as sim_output:
            df = do_extract_interventions(sim_output, contagion)
        return df

    def extract_transmissions(
        self, output_file: Path, contagion_name: str = ""
    ) -> pl.DataFrame:
        contagion = find_contagion(contagion_name, self.ast1)
        with h5.File(output_file, "r") as sim_output:
            df = do_extract_transmissions(sim_output, contagion)
        return df
