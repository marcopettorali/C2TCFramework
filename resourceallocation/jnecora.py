# Copyright (C) 2025  Marco Pettorali

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


import copy
import itertools
import json
import multiprocessing as mp
import os
import pickle
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

import networkx as nx

from networking.entities import Host, Link, Process
from resourceallocation.context import Context, load_context
from utils.distribution import Distribution
from utils.dynamic_execute import dynamic_execute
from utils.ga import ga_optimization
from utils.plotting import draw_paths, draw_topology
from utils.printing import print

# Check if OMP_NUM_THREADS is set to 1
if "OMP_NUM_THREADS" not in os.environ or os.environ["OMP_NUM_THREADS"] != "1":
    print(
        "The OMP_NUM_THREADS environment variable is not set to 1.\nUse 'export OMP_NUM_THREADS=1' to set it.\nThis is required to avoid issues with multiprocessing and OpenMP.",
        style="error",
    )

    exit(1)

# GLOBAL VARIABLES (WATCH OUT!)
PACKET_LOSS_MS = 2000
MAX_PROCESSES_PER_HOST = 8
MAX_MNS_PER_PROCESS = 13


def _find_paths(graph, source, target):
    """
    Finds all simple paths between a source and target node in the graph.

    Args:
        graph (networkx.Graph): The topology graph.
        source (str): The source node.
        target (str): The target node.

    Returns:
        list: A list of paths, where each path is represented as a list of edges with source, destination, and link info.
    """
    all_shortest_paths = list(nx.all_simple_paths(graph, source, target))

    # Convert paths to edges with labels
    all_shortest_links = [[{"src": u, "dest": v, "info": graph[u][v]} for u, v in zip(path, path[1:])] for path in all_shortest_paths]

    return all_shortest_links


def _compute_end_to_end_communication_delays(context: Context):
    """
    Computes the end-to-end communication delays between processes and hosts in the topology graph.

    Args:
        context (Context): The simulation context containing the topology graph.

    Returns:
        Context: The updated context with computed communication delays stored in `context.links["gamma_com"]`.
    """
    topology_graph = context.topology_graph

    context.links["gamma_com"] = {}

    processes = [node for node in topology_graph if topology_graph.nodes[node]["type"] == "process"]
    hosts = [node for node in topology_graph if topology_graph.nodes[node]["type"] == "host"]

    link_distributions = {}
    end_to_end_distributions = {}

    # compute all possible links
    for process, host in itertools.product(processes, hosts):
        paths = _find_paths(topology_graph, process, host)
        paths_distributions = []
        for path in paths:
            path_delay = Distribution.dirac_delta(0)
            probability = 1

            for link in path:
                if "probability" in link["info"]:
                    probability = probability * link["info"]["probability"]
                key = (link["src"], link["dest"])
                if key not in link_distributions:
                    link_distributions[key] = Link(f"{link['src']}_{link['dest']}", delay_distribution=link["info"]["desc"])
                path_delay = path_delay + link_distributions[key].delay_distribution.pdf
            paths_distributions.append((path_delay, probability))

        packet_loss = 1 - sum([path[1] for path in paths_distributions])

        end_to_end_distributions[(process, host)] = Distribution.combine(
            [path[0] for path in paths_distributions] + [Distribution.dirac_delta(PACKET_LOSS_MS)],
            [path[1] for path in paths_distributions] + [packet_loss],
        )

        context.links["gamma_com"][(process, host)] = end_to_end_distributions[(process, host)]

    return context


_DISABLE_CACHE = False


def disable_cache():
    """
    Disables the caching mechanism for delay computations.
    """
    global _DISABLE_CACHE
    _DISABLE_CACHE = True


_DELAY_CACHE = {}


def compute_delay_at_min_reliability(context: Context, process: Process, host: Host, cpu_share: float):
    """
    Computes the end-to-end delay at the min reliability percentile for a process on a host with a given CPU share.
    Returns the delay (in ms) at the min reliability percentile.
    Note that this method is agnostic to any existing allocation.
    """
    assert isinstance(process, Process), "process must be an instance of Process"
    assert isinstance(host, Host), "host must be an instance of Host"
    assert isinstance(cpu_share, (float, int)), f"cpu_share must be a float or a int, and is a {type(cpu_share)}"

    # ASSUMPTION: context is always the same, it is NEVER changed!
    # ASSUMPTION: the only params that affect the delay computation are the process, its mns, the host and the cpu_share
    # check if the delay has already been computed and if it is out-of-scale
    cache_key = (process.name, host.label, process.mns)
    if cache_key in _DELAY_CACHE:
        for k, v in _DELAY_CACHE[cache_key].items():
            if v >= PACKET_LOSS_MS and cpu_share <= k:
                # print(
                #     f"Already computed delay at min reliability for process {process.name} on host {host.label} with CPU share = {k}. Hence, now that CPU share is {cpu_share}, I am returning {PACKET_LOSS_MS}",
                #     style="warning",
                # )
                return PACKET_LOSS_MS

    # retrieve the communication delay to get to the BR when the MNs are directly connected
    gamma_com = context.links["gamma_com"][(process.name, host.label)]

    # print("Bisection job with scale", scale, style="debug")
    # compute the scaled execution delay (based on the host's CPU and the scale parameter)

    gamma_exe = (
        process.application.benchmark.distribution.pdf * ((process.application.benchmark.cpu_ghz / host.cpu_ghz) * (1 / cpu_share))
    ).normalize()

    # check if gamma_exe is out-of-scale
    if gamma_exe.percentile(1) >= PACKET_LOSS_MS:
        # print(
        #     f"Execution time is out-of-scale for process {process.name} on host {host.label}, returning a dirac delta at {PACKET_LOSS_MS}ms",
        #     style="debug",
        # )

        # store the result in the cache
        if cache_key not in _DELAY_CACHE:
            _DELAY_CACHE[cache_key] = {}
        _DELAY_CACHE[cache_key][cpu_share] = PACKET_LOSS_MS

        return PACKET_LOSS_MS

    if host.infinite_parallelism:
        # print(
        #     f"Host {host.label} has infinite parallelism, the processing delay is equal to the execution delay",
        #     style="debug",
        # )
        # the processing delay is equal to the execution delay
        gamma_proc = gamma_exe
    else:
        # print(
        #     f"Computing the queuing time for process {process.name} on host {host.label}",
        #     style="debug",
        # )
        # compute the queuing time based on the number of MNs and the execution delay
        gamma_que = dynamic_execute(
            host.qtime_dist_function, gamma_exe, process, host, cache_index=f"{process.name}_{host.label}_{cpu_share}"
        ).normalize()

        # compute the processing delay (queuing + execution)
        gamma_proc = gamma_que + gamma_exe

    # compute the total delay (communication + processing)
    gamma_tot: Distribution = gamma_com + gamma_proc
    gamma_tot = gamma_tot.normalize()

    # compute the delay at the min reliability percentile
    delay_at_min_reliability = gamma_tot.quantile(process.min_reliability)

    # TODO remove 1.5 ms for numerical errors
    delay_at_min_reliability -= 1.5

    # print(f"Delay at min reliability = {delay_at_min_reliability} ms", style="debug")

    # store the result in the cache
    if not _DISABLE_CACHE and cache_key not in _DELAY_CACHE:
        _DELAY_CACHE[cache_key] = {}
    if not _DISABLE_CACHE:
        _DELAY_CACHE[cache_key][cpu_share] = delay_at_min_reliability

    # return the delay at the min reliability percentile
    return delay_at_min_reliability


def _worker(process: Process, host: Host, nmns, context: Context):
    """
    Worker function for multiprocessing to compute delays for a process on a host.

    Args:
        process (Process): The process to compute delays for.
        host (Host): The host where the process is allocated.
        nmns (int): The number of MNs (Mobile Nodes) allocated to the process.
        context (Context): The simulation context.

    Returns:
        list: A list of tuples containing process name, host label, CPU share, MNs, and delay.
    """
    temp_process = copy.deepcopy(process)
    temp_process.mns = nmns

    max_delay_ms = temp_process.max_delay_ms

    results = []
    to_break = False
    for share in [1 / i for i in range(1, MAX_PROCESSES_PER_HOST + 1)]:
        if to_break:
            results.append((temp_process.name, host.label, share, nmns, float("inf")))
            continue

        delay_at_rel = float(compute_delay_at_min_reliability(context, temp_process, context.hosts[host.label], share))
        # print(f"Process {process.name} on host {host.label} with {share*100}% share: {delay_at_rel}")
        results.append((temp_process.name, host.label, share, nmns, delay_at_rel))
        if delay_at_rel > max_delay_ms:
            to_break = True
            results.append((temp_process.name, host.label, share, nmns, float("inf")))

    return results


def _compute_gamma_tot_for_each_process_host_cpu_share(context: Context):
    """
    Precomputes the total delay (gamma_tot) for each process-host-CPU share combination.

    Args:
        context (Context): The simulation context.

    Returns:
        Context: The updated context with precomputed gamma_tot values.
    """
    context.links["gamma_tot_precomputed"] = {}

    cpu_share: float

    with mp.Pool(mp.cpu_count()) as pool:
        results = list(
            pool.starmap(
                _worker,
                tqdm(
                    [
                        (process, host, nmns, context)
                        for process, host, nmns in itertools.product(
                            context.processes.values(),
                            context.hosts.values(),
                            range(1, MAX_MNS_PER_PROCESS + 1),
                        )
                    ],
                    total=len(context.processes) * len(context.hosts) * MAX_MNS_PER_PROCESS,
                    desc="Computing gamma_tot",
                ),
            ),
        )

    # Flatten the list of results
    results = [item for sublist in results for item in sublist]

    print(f"Inserting {len(results)} results in the context", style="debug")
    for process_name, host_label, cpu_share, nmns, gamma_tot in results:
        context.links["gamma_tot_precomputed"][(process_name, host_label, cpu_share, nmns)] = float(gamma_tot)

    return context


class JNecora:
    def __init__(self):
        """
        Initializes the JNecora class.
        """
        self.context = None

    def set_context(self, context):
        """
        Sets the simulation context.

        Args:
            context (Context): The simulation context.
        """
        assert isinstance(context, Context), "context must be an instance of Context"
        self.context = context
        print(f"Context set", style="debug")

    @staticmethod
    def load_context_from_file(config_path: str, pickle_context: bool = True):
        """
        Loads the simulation context from a configuration file.

        Args:
            config_path (str): Path to the configuration file.
            pickle_context (bool): Whether to pickle the context for future use.

        Returns:
            Context: The loaded simulation context.
        """
        print(f"Importing context from file {config_path}", style="debug")

        # convert the config_path to a Path object
        config_path = Path(config_path)

        # check if in the parent folder there is a pickles folder and inside a pickle file with the same name as the config file
        pickle_path = config_path.parent / "pickles" / "jnecora" / f"{config_path.stem}.pkl"
        if pickle_path.exists():

            # unpickle the context and check if the extracted context.config_file_content is the same as the content of config file
            with open(pickle_path, "rb") as f:
                pickled_context = pickle.load(f)

            # load the config file
            with open(config_path, "r") as f:
                config = json.load(f)

            # check if the config file content is the same as the pickled context
            if pickled_context.config_file_content != config:
                raise Exception(
                    f"The content of the pickled context { pickle_path} is different from the content of the config file {config_path}! Exiting"
                )

            # print the date of last modification of the pickle file
            last_modified_time = datetime.fromtimestamp(pickle_path.stat().st_mtime)
            formatted_time = last_modified_time.strftime("%H:%M:%S, %A %d %B %Y")

            print(
                f"Found pickle file {pickle_path}, edited on {formatted_time}. Loading the context from the pickle file",
                style="warning",
            )

            # return the pickled context
            return pickled_context

        # load the config file
        context = load_context(config_path)

        # compute the gamma_com for each process and host
        context = _compute_end_to_end_communication_delays(context)

        # compute the gamma_proc for each process, host and cpu_share
        context = _compute_gamma_tot_for_each_process_host_cpu_share(context)

        # if pickle_context is True, pickle the context
        if pickle_context:
            # create the pickles folder if it does not exist
            pickles_folder = config_path.parent / "pickles" / "jnecora"
            pickles_folder.mkdir(exist_ok=True, parents=True)

            # pickle the context
            with open(pickle_path, "wb") as f:
                pickle.dump(context, f)

            print(f"Pickled context to {pickle_path}", style="warning")

        return context

    def import_context_from_file(self, config_path: str, pickle_context: bool = True):
        """
        Imports the simulation context from a configuration file.

        Args:
            config_path (str): Path to the configuration file.
            pickle_context (bool): Whether to pickle the context for future use.
        """
        self.set_context(JNecora.load_context_from_file(config_path, pickle_context))

    def _max_deltadelay_objective(self, solution):
        """
        Computes the objective function for maximizing delta delay.

        Args:
            solution (list): A list of process-host assignments.

        Returns:
            float: The computed objective value.
        """
        sum_delta = 0
        for process_host in solution:
            process_name, host_label = process_host
            n_processes_on_host = sum([1 for elem in solution if elem[1] == host_label])
            gamma_tot = self.context.links["gamma_tot_precomputed"][
                (process_name, host_label, 1 / n_processes_on_host, self.context.processes[process_name].mns)
            ]

            delta = self.context.processes[process_name].max_delay_ms - gamma_tot

            sum_delta += delta

        return sum_delta / len(solution)

    def _max_mns_objective(self, solution):
        """
        Computes the objective function for maximizing the number of MNs.

        Args:
            solution (list): A list of process-host assignments.

        Returns:
            float: The computed objective value.
        """
        # compute how many MNs can be allocated to each process
        sum_mns = 0
        for process_name, host_label in solution:
            if self.context.hosts[host_label].infinite_parallelism:
                cpu_share = 1
            else:
                cpu_share = 1 / (sum([1 for elem in solution if elem[1] == host_label]))

            max_delay_ms = self.context.processes[process_name].max_delay_ms
            valid_nms = 0
            for nmns in range(1, MAX_MNS_PER_PROCESS + 1):
                gamma_tot = context.links["gamma_tot_precomputed"][(process_name, host_label, cpu_share, nmns)]
                if gamma_tot <= max_delay_ms:
                    valid_nms = nmns
                else:
                    break

            sum_mns += valid_nms

        return sum_mns

    def allocate_all_processes_optimal(self, objective="max_mns"):
        """
        Allocates all processes optimally based on the specified objective.

        Args:
            objective (str): The optimization objective ("max_mns" or "max_deltadelay").

        Returns:
            tuple: The best solution and its value.
        """
        if objective not in ["max_mns", "max_deltadelay"]:
            raise ValueError(f"Objective {objective} is not supported. Supported objectives are 'max_mns' and 'max_deltadelay'")

        process_names = list(self.context.processes.keys())
        host_labels = list(self.context.hosts.keys())

        # store the min CPU share required to allocate each process on each host
        data = {}
        for p, h, c, m in self.context.links["gamma_tot_precomputed"]:
            if m == 1 and self.context.links["gamma_tot_precomputed"][(p, h, c, m)] <= self.context.processes[p].max_delay_ms:
                data.setdefault((p, h), []).append(c)

        min_cpu_share_per_process_host = {
            (p, h): min(data.get((p, h), [float("inf")])) for p, h in itertools.product(process_names, host_labels)
        }

        # check if some process can be allocated to a host with infinite parallelism
        # if so, allocate it to the first host with the minimum CPU share
        init_allocation = [
            (process_name, min(allocations_on_infparal_hosts, key=lambda x: x[1])[0])
            for process_name in process_names
            if (
                allocations_on_infparal_hosts := [
                    (host_label, min_cpu_share_per_process_host[(process_name, host_label)])
                    for host_label in host_labels
                    if min_cpu_share_per_process_host[(process_name, host_label)] != float("inf")
                    and self.context.hosts[host_label].infinite_parallelism
                ]
            )
        ]

        # remove processes in the init_allocation from the process_names list
        for process_name, _ in init_allocation:
            process_names.remove(process_name)

        print(f"Initial allocation: {init_allocation}", style="debug")
        print(f"Remaining processes: {process_names}", style="debug")

        solutions = []  # This will store all valid candidate solutions

        def recursive_assign(current_solution=[], current_process_index=0, hosts_utilization=None):
            hosts_utilization = hosts_utilization or {host_label: float("inf") for host_label in host_labels}

            # Base case: all processes have been assigned
            if current_process_index == len(process_names):
                assert all(u >= 0 for u in hosts_utilization.values())
                solutions.append(current_solution.copy())
                return

            # Try to assign the current process to each host
            current_process = process_names[current_process_index]
            for host_label in host_labels:
                # Compute how many processes can be allocated to the same host to guarantee QoS for the current process
                max_processes_same_host = round(1 / min_cpu_share_per_process_host[(current_process, host_label)])
                new_hosts_utilization = hosts_utilization.copy()
                # the new available utilization must satisfy the most strigent process allocated to this host
                new_hosts_utilization[host_label] = min(new_hosts_utilization[host_label], max_processes_same_host) - 1
                if new_hosts_utilization[host_label] >= 0:
                    new_solution = current_solution.copy()
                    new_solution.append((current_process, host_label))
                    recursive_assign(new_solution, current_process_index + 1, new_hosts_utilization)

        # Start the recursion from the first process
        recursive_assign(init_allocation)

        # Evaluate the solutions according to the objective function
        if len(solutions) == 0:
            return None, 0

        objective_function = self._max_deltadelay_objective if objective == "max_deltadelay" else self._max_mns_objective

        with mp.Pool(mp.cpu_count()) as pool:
            results = pool.map(objective_function, solutions)

        # Find the best solution (that maximizes the objective function)
        max_value = max(results)
        best_solution = solutions[results.index(max_value)]

        return best_solution, max_value

    def _objective_ga_wrapper(self, solution, function):
        # solution is an array with length len(process_names) and values in [0, len(host_labels) - 1]
        # convert it to a list of tuples (process_name, host_label)
        process_names = list(self.context.processes.keys())
        host_labels = list(self.context.hosts.keys())
        solution = [(process_names[i], host_labels[solution[i]]) for i in range(len(solution))]
        # call the _max_mns_objective function
        return function(solution)

    def allocate_all_processes_besteffort(self, objective="max_mns", method="exhaustive"):
        """
        Allocates all processes using a best-effort approach.

        Args:
            objective (str): The optimization objective ("max_mns" or "max_deltadelay").
            method (str): The allocation method ("exhaustive" or "ga").

        Returns:
            tuple: The best solution and its value.
        """
        process_names = list(self.context.processes.keys())
        host_labels = list(self.context.hosts.keys())

        if method == "exhaustive":

            # compute the number of possible combinations
            num_combinations = len(host_labels) ** len(process_names)

            if num_combinations <= 1000000:
                print(f"Exhaustive search with {num_combinations} combinations", style="debug")
            elif num_combinations <= 1000000000:
                print(
                    f"The search space size is {num_combinations / 1000000:.2f}M. This may take a while. Consider using non-exhaustive methods (e.g. 'ga').",
                    style="warning",
                )
            else:
                print(
                    f"The search space size is {num_combinations / 1000000000:.2f}B. PLEASE CONSIDER USING NON-EXHAUSTIVE METHODS (e.g. 'ga').",
                    style="error",
                )

            # Generate all possible combinations of process-host assignments
            all_combinations = [
                [(process_names[i], host_label) for i, host_label in enumerate(combination)]
                for combination in itertools.product(host_labels, repeat=len(process_names))
            ]

            objective_function = self._max_deltadelay_objective if objective == "max_deltadelay" else self._max_mns_objective

            with mp.Pool(mp.cpu_count()) as pool:
                results = pool.map(objective_function, all_combinations)

            # Find the best solution (that maximizes the objective function)
            best_value = max(results)
            best_solution = all_combinations[results.index(best_value)]

        elif method == "ga":
            print(f"Using GA with {len(process_names)} processes and {len(host_labels)} hosts", style="debug")

            objective_function = self._max_deltadelay_objective if objective == "max_deltadelay" else self._max_mns_objective

            ret = ga_optimization(
                fitness_function=self._objective_ga_wrapper,
                fitness_args=(objective_function,),
                vector_size=len(process_names),
                value_type=int,
                value_range=(0, len(host_labels) - 1),
                population_size=1000,
                num_processes=1,
            )

            # convert the best solution to a list of tuples (process_name, host_label)
            best_solution = [(process_names[i], host_labels[ret["best_solution"][i]]) for i in range(len(ret["best_solution"]))]
            best_value = ret["best_fitness"]

        else:
            raise ValueError(f"Method {method} is not supported. Supported methods are 'exhaustive' and 'ga'")

        return best_solution, best_value


if __name__ == "__main__":
    import itertools
    import sys
    import argparse
    import matplotlib.pyplot as plt

    # with argparse, the first parameter is the scenario name relative to configs/, there is a argument "result-path" that is the path of the json file where to save the results
    # there is a command --draw-route src dest that draws the path from src to dest and exits
    parser = argparse.ArgumentParser(description="JNecora: Resource allocation for C2TC")
    parser.add_argument("scenario_name", type=str, help="The scenario name relative to configs/")
    parser.add_argument("--result-path", type=str, default="results.json", help="The path of the json file where to save the results relative to out/")
    parser.add_argument(
        "--draw-route",
        type=str,
        nargs=2,
        help="Draw the route from src to dest and exit",
    )
    args = parser.parse_args()
    # check if the scenario name is provided
    if args.scenario_name is None:
        print("Scenario name is required", style="error")
        sys.exit(1)
    # check if the result path is provided
    if args.result_path is None:
        print("Result path is required", style="error")
        sys.exit(1)
    # check if the draw path is provided
    if args.draw_route is not None:
        src, dest = args.draw_route
        # check if src and dest are in the topology graph
        context = JNecora.load_context_from_file(f"configs/{args.scenario_name}.json")
        if src not in context.topology_graph.nodes:
            print(f"Node {src} is not in the topology graph", style="error")
            sys.exit(1)
        if dest not in context.topology_graph.nodes:
            print(f"Node {dest} is not in the topology graph", style="error")
            sys.exit(1)
        # draw the path from src to dest
        fig, ax = draw_topology(context.topology_graph)
        draw_paths(context.topology_graph, ax, src, dest)
        plt.show()
        sys.exit(0)

    # Load the context from the config file
    scenario_name = args.scenario_name
    context = JNecora.load_context_from_file(f"configs/{scenario_name}.json")

    jnecora = JNecora()
    jnecora.set_context(context)

    # Run the allocation (find an optimal solution)
    print(f"Allocating all processes with the optimal solution", style="info")

    ret = jnecora.allocate_all_processes_optimal()
    # ret = jnecora.allocate_all_processes_besteffort(method="ga")
    solution, value = ret

    # if the optimal solution is not found, try with the best-effort solution
    if not solution:
        print("No optimal solution found. Trying with the best-effort solution", style="warning")
        ret = jnecora.allocate_all_processes_besteffort()
        solution, value = ret

    # print the solution
    print(f"Best solution: {solution}", style="debug")
    print(f"Best value: {value}", style="debug")

    # put the allocated CPU share and the max number of MNs in the solution
    new_solution = []
    for all in solution:
        process_name, host_label = all
        max_delay_ms = context.processes[process_name].max_delay_ms

        if context.hosts[host_label].infinite_parallelism:
            cpu_share = 1
        else:
            cpu_share = 1 / (sum([1 for elem in solution if elem[1] == host_label]))

        valid_nms = 0
        for nmns in range(1, MAX_MNS_PER_PROCESS + 1):
            gamma_tot = context.links["gamma_tot_precomputed"][(process_name, host_label, cpu_share, nmns)]
            if gamma_tot <= max_delay_ms:
                valid_nms = nmns
            else:
                break

        if valid_nms == 0:
            host_label = "None"
            cpu_share = 0

        new_solution.append((process_name, host_label, cpu_share, valid_nms))

    results = [new_solution, value]
    print(new_solution)

    # save the results in a json file
    import json

    filename = "out/" + args.result_path

    import os

    # if the path does not exist, create it
    if not os.path.exists(filename):
        with open(filename, "w") as f:
            json.dump({scenario_name:results}, f, indent=4)
    else:

        with open(filename, "r") as f:
            data = json.load(f)
            data[scenario_name] = results

        with open(filename, "w") as f:
            json.dump(data, f, indent=4)
