from networking.entities import Process
from utils.plotting import draw_paths, draw_topology
from resourceallocation.jnecora import JNecora
from utils.logging import error, debug, info, warning, print
import copy


class ProcessSplit:
    def __init__(self, process: Process, parent_process_name: str):
        self.process = process
        self.parent_process_name = parent_process_name


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
            host_label: {"cpu": host.cpu_ghz, "ram": host.ram_gb} for host_label, host in self.context.hosts.items()
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
        if self.splitting_policy in ["no_splitting", "lazy_splitting"]:
            mns_to_allocate = process.mns
        elif self.splitting_policy == "greedy_splitting":
            mns_to_allocate = 1
        else:
            raise ValueError(f"Invalid splitting policy: {self.splitting_policy}")

        while mns_to_allocate > 0:
            debug(f"Trying to allocate {mns_to_allocate} MNs for process {process_name}")

            # retrieve the delay requirement of this process
            max_delay_ms = process.max_delay_ms

            # # Create a split for this process with the right number of MNs to allocate
            # _temp_process = copy.deepcopy(process)
            # _temp_process.mns = mns_to_allocate
            # current_split = ProcessSplit(_temp_process, process_name)

            # check if current split can be allocated on which host
            candidate_hosts = []
            for host_label, host in self.context.hosts.items():
                # retrieve the available CPU and RAM for this host
                _available_resources = self.available_resources_per_host[host_label]
                available_cpu, available_ram = _available_resources["cpu"], _available_resources["ram"]
                available_cpu_share = available_cpu / host.cpu_ghz if host.cpu_ghz > 0 else 0
                debug(f"Host {host_label} has {available_cpu} GHz ({available_cpu_share}) and {available_ram} GB available")

                # check if the current host already has a split for this process
                already_existing_split_on_host = None
                splits_on_host = self.process_splits_per_host.get(host_label, [])
                if process_name in [split.process_name for split in splits_on_host]:
                    already_existing_split_on_host = next(split for split in splits_on_host if split.process_name == process_name)
                    debug(f"Host {host_label} already has a split for process {process_name}: {already_existing_split_on_host}")
                else:
                    debug(f"Host {host_label} does not have a split for process {process_name}")

                # if there are no already existing splits, consider the RAM constraint
                # otherwise, the two splits will be merged, and the amount of RAM is equal to that of one split
                if already_existing_split_on_host is None:
                    # if available RAM is less than the process RAM, skip this host
                    if available_ram < process.application.ram_occupancy_gb:
                        debug(f"Host {host_label} does not have enough RAM for process {process_name} ({available_ram} < {process.ram_gb})")
                        continue

                # we now check the CPU constraint
                # we find the highest number of MNs that can be served using the available CPU on the host
                # (context.links["gamma_tot_precomputed"][(process_name, host_label, cpu_share, nmns)] = float(gamma_tot))
                best_fitting_cpu_share_record = max(
                    (
                        (p, h, c, m)
                        for (p, h, c, m), gamma_tot in self.context.links["gamma_tot_precomputed"].items()
                        if p == process_name and h == host_label and gamma_tot <= max_delay_ms and c <= available_cpu_share
                    ),
                    key=lambda x: x[3],
                    default=None,
                )

                # if the number of MNs that can be served is 0, skip this host
                if best_fitting_cpu_share_record is None:
                    debug(f"Host {host_label} cannot serve any MNs for process {process_name}")
                    continue

                # TODO CONTINUA QUI
                # num_mns_allocatable = best_fitting_cpu_share_record[3] if best_fitting_cpu_share_record is not None else 0

            return


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

    djnecora.add_process("P3")
