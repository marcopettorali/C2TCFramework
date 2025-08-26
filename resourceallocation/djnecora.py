from networking.entities import Process
from utils.plotting import draw_paths, draw_topology
from resourceallocation.jnecora import JNecora
from utils.logging import error, debug, info, warning, print
import copy
import random

# set constant random seed for reproducibility
random.seed(42)

class ProcessSplit:
    def __init__(self, process_name: str, num_mns: int, cpu_share: float):
        self.process_name = process_name
        self.num_mns = num_mns
        self.cpu_share = cpu_share

    def __repr__(self):
        return f"ProcessSplit(process={self.process_name}, mns={self.num_mns}, cpu_share={self.cpu_share})"


class DJNecora(JNecora):
    def __init__(self, splitting_policy, selection_policy):

        if splitting_policy not in ["no_splitting", "lazy_splitting", "greedy_splitting"]:
            raise ValueError(f"Invalid splitting policy: {splitting_policy}")

        if selection_policy not in ["first_fit", "next_fit", "best_fit", "worst_fit", "random"]:
            raise ValueError(f"Invalid selection policy: {selection_policy}")

        super().__init__()

        self.splitting_policy = splitting_policy
        self.selection_policy = selection_policy
        self._hosts_initialized = False

    @staticmethod
    def load_context_from_file(config_path: str, pickle_context: bool = True, pickle_folder_relative_path: str = "pickles/djnecora"):
        return JNecora.load_context_from_file(config_path, pickle_context, pickle_folder_relative_path)

    def initialize_hosts(self):
        if self.context is None:
            raise RuntimeError("Context is not set. Please, use set_context() first.")

        self._hosts_initialized = True

        # Initialize host resources
        self.available_resources_per_host = {
            host_label: {"cpu_share": 1, "ram": host.ram_gb} for host_label, host in self.context.hosts.items()
        }

        self.process_splits_per_host = {host_label: [] for host_label in self.context.hosts.keys()}

        info(f"{len(self.context.hosts)} **hosts initialized**")

    def add_process(self, process_name: str):
        # guards
        if not self._hosts_initialized:
            raise RuntimeError("Hosts have not been initialized. Please, use initialize_hosts() first.")

        if process_name not in self.context.processes:
            raise ValueError(f"Process {process_name} does not exist.")

        for host, process_list in self.process_splits_per_host.items():
            if process_name in process_list:
                raise ValueError(f"Process {process_name} is already added (host {host}).")

        process = self.context.processes[process_name]
        info(f"**Adding process** {process_name} with {process.mns} MNs")

        # iteratively try to add the process according to the splitting policy
        allocated_mns = 0
        while allocated_mns < process.mns:            
            if self.splitting_policy in ["no_splitting", "lazy_splitting"]:
                mns_to_allocate_in_iteration = process.mns - allocated_mns
            elif self.splitting_policy == "greedy_splitting":
                mns_to_allocate_in_iteration = 1
            else:
                raise ValueError(f"Invalid splitting policy: {self.splitting_policy}")

            debug(f"**MNs to allocate** in this iteration according to the splitting policy {self.splitting_policy}: {mns_to_allocate_in_iteration}")

            # retrieve the delay requirement of this process
            max_delay_ms = process.max_delay_ms

            # # Create a split for this process with the right number of MNs to allocate
            # _temp_process = copy.deepcopy(process)
            # _temp_process.mns = mns_to_allocate
            # current_split = ProcessSplit(_temp_process, process_name)

            # check if current split can be allocated on which host
            candidate_hosts = []
            for host_label, host in self.context.hosts.items():
                debug(f"\t--- **Host {host_label}** ---")
                # retrieve the available CPU and RAM for this host
                _available_resources = self.available_resources_per_host[host_label]
                available_cpu_share, available_ram = _available_resources["cpu_share"], _available_resources["ram"]
                debug(f"\tHost {host_label} has {available_cpu_share*100}% and {available_ram} GB available")

                # check if the current host already has a split for this process
                already_existing_split_on_host = None
                splits_on_host = self.process_splits_per_host.get(host_label, [])
                assert len(splits_on_host) <= 1, "There should be at most one split per process on each host"

                if process_name in [split.process_name for split in splits_on_host]:
                    already_existing_split_on_host = next(split for split in splits_on_host if split.process_name == process_name)
                    debug(f"\tHost {host_label} already has a split for process {process_name}: {already_existing_split_on_host}")
                else:
                    debug(f"\tHost {host_label} does not have a split for process {process_name}")

                # if there are no already existing splits, consider the RAM constraint
                # otherwise, the two splits will be merged, and the amount of RAM is equal to that of one split
                if already_existing_split_on_host is None:
                    # if available RAM is less than the process RAM, skip this host
                    if available_ram < process.application.ram_occupancy_gb:
                        debug(f"\tHost {host_label} does not have enough RAM for process {process_name} ({available_ram} < {process.ram_gb})")
                        continue

                # we now check the CPU constraint
                # we find the highest number of MNs that can be served using the available CPU on the host
                # (context.links["gamma_tot_precomputed"][(process_name, host_label, cpu_share, nmns)] = float(gamma_tot))
                best_fitting_cpu_share_record = max(
                    (
                        (p, h, c, m)
                        for (p, h, c, m), gamma_tot in self.context.links["gamma_tot_precomputed"].items()
                        if p == process_name
                        and h == host_label
                        and gamma_tot <= max_delay_ms
                        and c <= available_cpu_share
                        and m <= mns_to_allocate_in_iteration
                    ),
                    key=lambda x: (x[3], -x[2]),  # prefer higher number of MNs, then lower CPU share
                    default=None,
                )

                # if the number of MNs that can be served is 0, skip this host
                if best_fitting_cpu_share_record is None:
                    debug(f"\tHost {host_label} cannot serve any MNs for process {process_name}")
                    continue

                cpu_share, supported_mns = best_fitting_cpu_share_record[2:4]
                assert (
                    supported_mns > 0 and supported_mns <= mns_to_allocate_in_iteration
                ), "Number of MNs allocatable must be > 0 and <= mns_to_allocate"
                debug(f"\tHost {host_label} can serve {supported_mns} MNs for process {process_name} with CPU share {cpu_share*100}%")

                # add this host to the candidate hosts
                candidate_hosts.append({"host_label": host_label, "supported_mns": supported_mns, "cpu_share": cpu_share})

            debug(f"--- **All hosts analyzed** ---")
            # if the splitting policy is no_splitting, we can only allocate if there is a host that can serve all the MNs
            # if the splitting policy is lazy_splitting, we only consider hosts that maximize the splitting size (i.e. maximize supported_mns) to minimize the number of splits
            if self.splitting_policy == "no_splitting":
                eliminated_hosts = [c for c in candidate_hosts if c["supported_mns"] < mns_to_allocate_in_iteration]
                candidate_hosts = [c for c in candidate_hosts if c["supported_mns"] == mns_to_allocate_in_iteration]
            elif self.splitting_policy == "lazy_splitting":
                eliminated_hosts = [c for c in candidate_hosts if c["supported_mns"] < max([c["supported_mns"] for c in candidate_hosts])]
                candidate_hosts = [c for c in candidate_hosts if c["supported_mns"] == max([c["supported_mns"] for c in candidate_hosts])]
            elif self.splitting_policy == "greedy_splitting":
                eliminated_hosts = []

            if len(eliminated_hosts) > 0:
                debug(f"According to the **{self.splitting_policy}** splitting policy, the following candidates have been eliminated: {eliminated_hosts}")
            else:
                debug(f"No candidates have been eliminated according to the **{self.splitting_policy}** splitting policy.")

            # if there is one at least one candidate with infinite parallelism, select it
            infinite_parallelism_candidates = [ch for ch in candidate_hosts if self.context.hosts[ch["host_label"]].infinite_parallelism]
            if len(infinite_parallelism_candidates) > 0:
                candidate_hosts = infinite_parallelism_candidates
                debug(f"The following candidates with **infinite parallelism** are preferred: {infinite_parallelism_candidates}.")
            else:
                debug("There are no candidates with **infinite parallelism**.")


            # if there are no candidates after the filtering, the process cannot be allocated
            if len(candidate_hosts) == 0:
                info(f"There are no candidates for process {process_name} (remaining {mns_to_allocate_in_iteration} MNs)")
                return

            # otherwise let the selection policy choose one of the candidate hosts
            debug("--- **Applying selection policy** ---")
            debug(f"Candidate hosts after filtering: {candidate_hosts}")
            
            selected_host = None
            if self.selection_policy == "first_fit":
                selected_host = candidate_hosts[0]
            if self.selection_policy == "next_fit":
                selected_host = candidate_hosts[1] if len(candidate_hosts) > 1 else candidate_hosts[0]
            if self.selection_policy == "best_fit":
                selected_host =  min(
                candidate_hosts,
                key=lambda x: self.available_resources_per_host[x["host_label"]]["cpu_share"] - x["cpu_share"]
            )  
            if self.selection_policy == "worst_fit":
                selected_host =  max(
                candidate_hosts,
                key=lambda x: self.available_resources_per_host[x["host_label"]]["cpu_share"] - x["cpu_share"]
            )  
            if self.selection_policy == "random":
                selected_host = random.choice(candidate_hosts)
            
            # allocate the split on the selected host
            process_split = ProcessSplit(process_name, selected_host["supported_mns"], selected_host["cpu_share"])
            self.process_splits_per_host[selected_host["host_label"]].append(process_split)
            debug(self.process_splits_per_host)
            debug(f"Selected host: {selected_host} according to the **{self.selection_policy}** selection policy")

            # update the available resources on the selected host
            self.available_resources_per_host[selected_host["host_label"]]["cpu_share"] -= selected_host["cpu_share"] if not self.context.hosts[selected_host["host_label"]].infinite_parallelism else 0
            self.available_resources_per_host[selected_host["host_label"]]["ram"] -= process.application.ram_occupancy_gb
            debug(f"Updated available resources on host {selected_host['host_label']}: {self.available_resources_per_host[selected_host['host_label']]}")

            # update the number of allocated MNs
            allocated_mns += selected_host["supported_mns"]
            debug(f"Allocated MNs for process {process_name}: {allocated_mns}/{process.mns} MNs")


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
    parser.add_argument("selection_policy", type=str, help="The selection policy to use: first_fit, next_fit, best_fit, worst_fit, random")
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

    djnecora.add_process("P1")
