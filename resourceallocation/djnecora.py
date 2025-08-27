import numpy as np
from utils.plotting import draw_paths, draw_topology
from resourceallocation.jnecora import JNecora
from utils.logging import error, debug, info
import random
from dataclasses import dataclass

# set constant random seed for reproducibility
random.seed(42)


@dataclass
class ProcessSplit:
    process_name: str
    num_mns: int
    cpu_share: float

    def __repr__(self):  # keep your readable repr
        return f"{self.process_name}({self.num_mns}/{self.cpu_share*100:.2f}%)"


class DJNecora(JNecora):
    def __init__(self, splitting_policy, selection_policy):
        if splitting_policy not in {"no_splitting", "lazy_splitting", "greedy_splitting"}:
            raise ValueError(
                f"Invalid splitting policy: {splitting_policy}.\nPlease choose from: no_splitting, lazy_splitting, greedy_splitting"
            )
        if selection_policy not in {"first_fit", "next_fit", "best_fit", "worst_fit", "random_fit"}:
            raise ValueError(
                f"Invalid selection policy: {selection_policy}.\nPlease choose from: first_fit, next_fit, best_fit, worst_fit, random_fit"
            )
        super().__init__()
        self.splitting_policy = splitting_policy
        self.selection_policy = selection_policy
        self._hosts_initialized = False

    @staticmethod
    def load_context_from_file(
        config_path: str,
        pickle_context: bool = True,
        pickle_folder_relative_path: str = "pickles/djnecora",
        _cpu_shares=list([float(x) for x in np.linspace(0, 1, 51)][1:]),
    ):
        return JNecora.load_context_from_file(config_path, pickle_context, pickle_folder_relative_path, _cpu_shares=_cpu_shares)

    def initialize_hosts(self):
        if self.context is None:
            raise RuntimeError("Context is not set. Please, use set_context() first.")

        self._hosts_initialized = True
        self._available_resources_per_host = {
            h: {"cpu_share": (1 if not host.infinite_parallelism else float("inf")), "ram": host.ram_gb}
            for h, host in self.context.hosts.items()
        }
        self._allocation_table_per_host = {h: [] for h in self.context.hosts}
        info(f"{len(self.context.hosts)} **hosts initialized**")

    # ---------- small helpers (compact + shared logic) ----------

    def _get_split_on_host(self, process_name: str, host_label: str) -> ProcessSplit | None:
        # at most one split per process/host (as in your assert)
        return next((x for x in self._allocation_table_per_host.get(host_label, []) if x.process_name == process_name), None)

    def _best_fit_record(
        self, process_name: str, host_label: str, max_delay_ms: float, available_cpu_share: float, need_mns: int, split: ProcessSplit | None
    ):
        best_fit_record = max(
            (
                (p, h, c, m)
                for (p, h, c, m), g in self.context.links["gamma_tot_precomputed"].items()
                if p == process_name
                and h == host_label
                and g <= max_delay_ms
                and c <= available_cpu_share + (split.cpu_share if split else 0)
                and m <= need_mns + (split.num_mns if split else 0)
                and m >= (split.num_mns if split else 1)
            ),
            key=lambda x: (x[3], -x[2]),  # max supported MNs, min CPU share
            default=None,
        )

        if not best_fit_record:
            return None

        cpu_share, mns = best_fit_record[2], best_fit_record[3]

        # If there was a split, we return the difference
        if split:
            cpu_share -= split.cpu_share
            mns -= split.num_mns

            if mns == 0:
                return None

        assert 0 < mns <= need_mns, "Number of MNs allocatable must be > 0 and <= mns_to_allocate"
        return {"host_label": host_label, "supported_mns": mns, "cpu_share": cpu_share, "_had_split": bool(split)}

    def _filter_by_splitting_policy(self, candidates: list[dict], needed_mns: int):
        if not candidates:
            return []
        if self.splitting_policy == "no_splitting":
            return [c for c in candidates if c["supported_mns"] == needed_mns]
        if self.splitting_policy == "lazy_splitting":
            m = max(c["supported_mns"] for c in candidates)
            return [c for c in candidates if c["supported_mns"] == m]
        return candidates  # greedy_splitting

    def _prefer_infinite_parallelism(self, candidates: list[dict]):
        inf = [c for c in candidates if self.context.hosts[c["host_label"]].infinite_parallelism]
        return inf or candidates

    def _selection_policy(self, candidates: list[dict]):
        key = lambda x: self._available_resources_per_host[x["host_label"]]["cpu_share"] - x["cpu_share"]
        policy = {
            "first_fit": lambda cands: cands[0],
            "next_fit": lambda cands: cands[1] if len(cands) > 1 else cands[0],
            "best_fit": lambda cands: min(cands, key=key),
            "worst_fit": lambda cands: max(cands, key=key),
            "random_fit": lambda cands: random.choice(cands),
        }[self.selection_policy]
        return policy(candidates)

    def _apply_allocation(self, process_name: str, selected: dict, process) -> bool:
        host = selected["host_label"]
        split = self._get_split_on_host(process_name, host)

        # If there was a split, we need to merge it
        if split:
            split.num_mns += selected["supported_mns"]
            split.cpu_share += selected["cpu_share"]
            self._available_resources_per_host[host]["cpu_share"] -= selected["cpu_share"]
            debug(f"Merged with existing split on host {host}: {split}")
            return False  # No new split created
        
        # If there was no split, we create a new one
        ps = ProcessSplit(process_name, selected["supported_mns"], selected["cpu_share"])
        self._allocation_table_per_host[host].append(ps)
        self._available_resources_per_host[host]["cpu_share"] -= selected["cpu_share"]
        self._available_resources_per_host[host]["ram"] -= process.application.ram_occupancy_gb
        debug(f"Allocated new split on host {host}: {ps}")
        return True  # New split created

    # ---------- main API ----------

    def add_process(self, process_name: str):
        if not self._hosts_initialized:
            raise RuntimeError("Hosts have not been initialized. Please, use initialize_hosts() first.")
        if process_name not in self.context.processes:
            raise ValueError(f"Process {process_name} does not exist.")
        if any(process_name in lst for lst in self._allocation_table_per_host.values()):
            raise ValueError(f"Process {process_name} is already added on some host.")

        process = self.context.processes[process_name]
        info(f"**Adding process** {process_name} with {process.mns} MNs")

        allocated_mns, num_splits = 0, 0
        while allocated_mns < process.mns:
            mns_to_allocate = process.mns - allocated_mns if self.splitting_policy != "greedy_splitting" else 1
            debug(f"**MNs to allocate** this iteration ({self.splitting_policy}): {mns_to_allocate}")
            max_delay_ms = process.max_delay_ms

            # Iterate over all hosts to find suitable candidates
            candidates = []
            for h, host in self.context.hosts.items():
                debug(f"\t--- **Host {h}** ---")
                
                available = self._available_resources_per_host[h]
                debug(f"\tAvailable resources: {available}")

                split = self._get_split_on_host(process_name, h)
                debug(f"\tExisting split: {split}" if split else f"\tNo existing split for {process_name} on host {h}")

                # RAM check only when creating a new split
                if not split and available["ram"] < process.application.ram_occupancy_gb:
                    debug(f"\tNot enough RAM available on host {h} for process {process_name}")
                    continue

                # Compute the maximum MNs that can be allocated with the available CPU
                rec = self._best_fit_record(process_name, h, max_delay_ms, available["cpu_share"], mns_to_allocate, split)
                if rec:  # {host_label, supported_mns, cpu_share, _had_split}
                    candidates.append({k: rec[k] for k in ("host_label", "supported_mns", "cpu_share")})
                    debug(f"\tHost {h} is a candidate: {candidates[-1]}")
                else:
                    debug(f"\tHost {h} cannot be a candidate")

            debug("--- **Candidates evaluation** ---")
            candidates = self._filter_by_splitting_policy(candidates, mns_to_allocate)
            debug(f"Filtered candidates by splitting policy ({self.splitting_policy}): {candidates}")
            
            candidates = self._prefer_infinite_parallelism(candidates)
            debug(f"Candidates after preferring infinite parallelism: {candidates}")

            if not candidates:
                info(f"There are no candidates for process {process_name} (remaining {mns_to_allocate} MNs)")
                break

            # Apply selection policy
            selected = self._selection_policy(candidates)
            debug(f"Selected candidate by {self.splitting_policy} for process {process_name}: {selected}")

            # Allocate the resources, and in case create a new split
            new_split = self._apply_allocation(process_name, selected, process)
            allocated_mns += selected["supported_mns"]
            num_splits += int(new_split)
            debug(f"Updated available resources on host {selected['host_label']}: {self._available_resources_per_host[selected['host_label']]}")
            debug(f"Allocated MNs for process {process_name}: {allocated_mns}/{process.mns} MNs\n")

        debug(f"--- **Process {process_name} completed** ---")
        debug(f"Process allocation table: {self._allocation_table_per_host}")
        debug(f"Available resources: {self._available_resources_per_host}")
        info(f"**Process {process_name} allocation complete**: allocated {allocated_mns}/{process.mns} MNs, {num_splits} splits created")

    def add_1_mn_to_process(self, process_name: str):
        if process_name not in self.context.processes:
            raise ValueError(f"Process {process_name} does not exist.")
        if not any(self._get_split_on_host(process_name, h) for h in self.context.hosts):
            raise ValueError(f"Process {process_name} was not allocated before. Use allocate_process() to allocate it.")

        info(f"**Adding 1 MN** to process {process_name}")
        process = self.context.processes[process_name]
        max_delay_ms = process.max_delay_ms

        with_split, without_split = [], []
        for h, host in self.context.hosts.items():
            debug(f"\t--- Evaluating host {h} ---")

            avail = self._available_resources_per_host[h]
            debug(f"\tAvailable resources: {avail}")

            split = self._get_split_on_host(process_name, h)
            debug(f"\tExisting split on host {h}: {split}")

            # RAM needed only if creating a new split
            if not split and avail["ram"] < process.application.ram_occupancy_gb:
                debug(f"\tHost {h} cannot accommodate new split for process {process_name}")
                continue
            
            rec = self._best_fit_record(process_name, h, max_delay_ms, avail["cpu_share"], 1, split)
            if not rec:
                continue

            # Record found, add to appropriate list, according to whether it had a split or not
            (with_split if rec["_had_split"] else without_split).append({k: rec[k] for k in ("host_label", "supported_mns", "cpu_share")})
            debug(f"\tRecorded {'with' if rec['_had_split'] else 'without'} split for host {h}: {rec}")

        debug("--- **Candidates evaluation** ---")

        if self.splitting_policy == "no_splitting":
            candidates = with_split
            debug(f"Splitting policy: {self.splitting_policy}, Candidates (with split only): {candidates}")
        elif self.splitting_policy == "lazy_splitting":
            candidates = with_split or without_split
            debug(f"Splitting policy: {self.splitting_policy}, Candidates (lazy splitting): {candidates}")
        else:  # greedy
            candidates = with_split + without_split
            debug(f"Splitting policy: {self.splitting_policy}, Candidates (greedy): {candidates}")

        candidates = self._prefer_infinite_parallelism(candidates)
        debug(f"Candidates after preferring infinite parallelism: {candidates}")
        if not candidates:
            info(f"No candidate hosts available for adding 1 MN to process {process_name}")
            return

        # Select the best candidate
        selected = self._selection_policy(candidates)
        debug(f"Selected candidate by {self.selection_policy} for process {process_name}: {selected}")

        # Apply the allocation
        new_split = self._apply_allocation(process_name, selected, process)
        debug(f"Updated available resources on host {selected['host_label']}: {self._available_resources_per_host[selected['host_label']]}")
        info(f"--- **1 MN added** for process {process_name} on host {selected['host_label']} {'' if not new_split else f'(new split created)'} ---")
        debug(f"Process allocation table: {self._allocation_table_per_host}")
        debug(f"Available resources: {self._available_resources_per_host}")

    def __str__(self):
        at = ",\n".join(f'\t\t"{h}": {pls}' for h, pls in self._allocation_table_per_host.items())
        ar = ",\n".join(f'\t\t"{h}": {res}' for h, res in self._available_resources_per_host.items())
        return f'\n{{\n\t"allocation_table": {{\n{at}\n\t}}\n\t"available_resources": {{\n{ar}\n\t}}\n}}'


if __name__ == "__main__":
    import itertools
    import sys
    import argparse
    from argparse import RawTextHelpFormatter
    import matplotlib.pyplot as plt

    # with argparse, the first parameter is the scenario name relative to configs/, there is a argument "result-path" that is the path of the json file where to save the results
    # there is a command --draw-route src dest that draws the path from src to dest and exits
    parser = argparse.ArgumentParser(
        description='DJ-NECORA: Dynamic resource allocation for C2TC (2024, Marco Pettorali)\nM. Pettorali, F. Righetti, C. Vallati, S. K. Das and G. Anastasi, "Dynamic Resource Allocation in Cloud-to-Things Continuum for Real-Time IoT Applications," 2025 IEEE International Conference on Smart Computing (SMARTCOMP), Cork, Ireland, 2025, pp. 432-437, doi: 10.1109/SMARTCOMP65954.2025.00107.\nhttps://ieeexplore.ieee.org/document/11058665',
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument("scenario_name", type=str, help="The scenario name relative to configs/")
    parser.add_argument("splitting_policy", type=str, help="The splitting policy to use: no_splitting, lazy_splitting, greedy_splitting")
    parser.add_argument(
        "selection_policy", type=str, help="The selection policy to use: first_fit, next_fit, best_fit, worst_fit, random_fit"
    )
    parser.add_argument(
        "--result-path", type=str, default="results.json", help="The path of the json file where to save the results relative to out/"
    )
    parser.add_argument(
        "--draw-route",
        type=str,
        nargs=2,
        help="Draw the route from src to dest and exit",
    )
    args = parser.parse_args()
    # check if the scenario name is provided
    if args.scenario_name is None:
        error("Scenario name is required")
        sys.exit(1)
    # check if the result path is provided
    if args.result_path is None:
        error("Result path is required")
        sys.exit(1)
    # check if the draw path is provided
    if args.draw_route is not None:
        src, dest = args.draw_route
        # check if src and dest are in the topology graph
        context = JNecora.load_context_from_file(f"configs/{args.scenario_name}.json")
        if src not in context.topology_graph.nodes:
            error(f"Node {src} is not in the topology graph")
            sys.exit(1)
        if dest not in context.topology_graph.nodes:
            error(f"Node {dest} is not in the topology graph")
            sys.exit(1)
        # draw the path from src to dest
        fig, ax = draw_topology(context.topology_graph)
        draw_paths(context.topology_graph, ax, src, dest)
        plt.show()
        sys.exit(0)

    # Load the context from the config file
    context = DJNecora.load_context_from_file(f"configs/{args.scenario_name}.json")

    djnecora = DJNecora(args.splitting_policy, args.selection_policy)
    djnecora.set_context(context)
    djnecora.initialize_hosts()

    djnecora.add_process("P0")
    for i in range(6):
        djnecora.add_1_mn_to_process("P0")

    info(djnecora)
