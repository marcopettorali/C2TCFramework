"""
MOERA solves this optimization problem at each time slot:
    min     E_O + E_Q + E_R + E_M
    s.t.    \\sum_{s} x_{s,u,t} >= \\lambda_u
            \\sum_{u} x_{s,u,t} <= C_s
            x_{s,u,t} >= 0

where
    x       =   amount of CPU allocated for an app on a node
    E_O     =   Operation cost = energy OR CPU usage OR mantainance cost
    E_Q     =   Service quality cost = network delay. It includes routing from access node to the node hosting the app
    E_R     =   Reconfiguration cost = cost associated with incresing CPU usage on nodes (e.g. powering up a new server)
    E_M     =   Migration cost = cost associated with migrating an app from one node to another
    \\lambda =   CPU demand of an app
    C       =   CPU capacity of a node

Adaptations we made to MOERA to compare it with DJ-NECORA:
- 1 user = 1 MN
- MOERA executes at each time slot ==> every minute. This means that in some time slots nothing changes, while in others new apps/MNs arrive.
  This behavior is similar to DJ-NECORA, which is triggered by events (new app/MN arrival) when the time slot is very short (1 minute should be ok).
- Adding is not explicitly supported in MOERA. However, we can assume that each user requires lambda = 0 until a timeslot t', then requires lambda = X from t' onwards.

- For E_O, we use CPU usage for fair comparison with DJ-NECORA
  Hence E_O = \\sum_{s,u} x_{s,u,t}
- For E_Q, MOERA assumes delays to be constant, and only considers network delays.
  Hence, we use the **average delay** of the distribution we use for DJ-NECORA.
  Moreover, MOERA does not consider packet loss.
- For E_R, we set it to 0 (we simply do not consider it in DJ-NECORA)
- For E_M, we set it to 0 (we simply do not consider it in DJ-NECORA).
- Moreover, since we do not consider migration, once an app is placed on a node, it will not be moved.
  Hence, we add another constraint:
    x_{s,u,t} = x_{s,u,t-1} \\forall s,u,t
- Similarly to DJ-NECORA, the task of 1 MN is served by a single edge node
  Hence, we add another constraint:
    \\exists! s' : x_{s',u,t} > 0 \\forall u,t

- MOERA needs the positions of the MNs at each time slot, and does not assume to have any AOIs/future positions in advance.
  Since we do not consider migration, we also consider a static scenario.
  Otherwise, mobility without migration can be very limiting for MOERA, since once an app is placed on a node, it will not be moved, even if the MN moves far away from the node.

- Since 1 user = 1 MN, we set the CPU demand of an app (\\lambda) to be the minimum CPU GHz required to satisfy the app's latency requirement with the required reliability for a single MN.
  In this computation, we consider:
    - network delay = average network delay of the distribution we use for DJ-NECORA, averaged over all possible locations of the MN/edge node
    - no queuing delay (we are allocating only 1 MN)
    - execution time is variabile. We compute the CPU GHz required to satisfy the latency requirement with the required reliability, considering only the variability of the execution time.
"""

import itertools
from networking.entities import Link
from resourceallocation.context import Context
from resourceallocation.utils import find_paths
from utils.logging import debug, info


def _compute_average_end_to_end_communication_delays(context: Context):
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
        paths = find_paths(topology_graph, process, host)
        average_path_delay = 0
        for path in paths:
            path_delay = 0  # ms
            probability = 1

            for link in path:
                if "probability" in link["info"]:
                    probability = probability * link["info"]["probability"]
                key = (link["src"], link["dest"])
                if key not in link_distributions:
                    link_distributions[key] = Link(f"{link['src']}_{link['dest']}", delay_distribution=link["info"]["desc"])
                # extract the average delay
                link_avg_delay = link_distributions[key].delay_distribution.pdf.mean_value()
                path_delay = path_delay + link_avg_delay
            average_path_delay += path_delay * probability

        # store the average delay distribution
        context.links["gamma_com"][(process, host)] = float(average_path_delay)
    return context


def exponential_search(func, target_func_value, tolerance=0.001):
    """
    Performs an exponential search to find an approximate solution to the equation func(x) = target_value.

    Args:
        func (callable): The function for which we want to find the input value that produces the target output.
        target_func_value (float): The target output value we want to achieve.
        tolerance (float): The acceptable difference between func(x) and target_value to consider the search successful.
    Returns:
        float: The input value x such that func(x) is approximately equal to target_value within the specified tolerance.
    """
    old_x = 0
    x = 0.0001
    while func(x) > target_func_value:
        old_x = x
        x *= 2

    # Now perform a binary search between old_x and x
    low = x
    high = old_x

    while low <= high:
        mid = (low + high) / 2
        mid_value = func(mid)

        if abs(mid_value - target_func_value) <= tolerance:
            return mid
        elif mid_value < target_func_value:
            high = mid
        else:
            low = mid

    return (low + high) / 2  # Return the best estimate if exact match not found


class MOERA:
    def __init__(self):
        self.context = None
        self._allocation_map = {}  # {host: [(process, allocated_cpu)]}
        self._process_CPU_demand = {}  # {process: cpu_demand}

    def set_context(self, context: Context):
        self.context = context
        self.context = _compute_average_end_to_end_communication_delays(self.context)

    def add_1_mn(self, process_name: str):
        if process_name not in self.context.processes:
            raise ValueError(f"Process {process_name} does not exist.")

        process = self.context.processes[process_name]

        # compute CPU demand of the process
        if process_name not in self._process_CPU_demand:
            min_cpu_ghz = None
            # network delay = average network delay between process and host
            avg_network_delay = 0
            for host_name, host in self.context.hosts.items():
                avg_network_delay += self.context.links["gamma_com"][(process_name, host_name)]
            avg_network_delay = avg_network_delay / len(self.context.hosts)

            # compute the minimum CPU GHz required to satisfy the latency requirement with the required reliability for a single MN
            gamma_func = lambda cpu_ghz: (
                float("inf")
                if cpu_ghz == 0
                else (process.application.benchmark.distribution.pdf * ((process.application.benchmark.cpu_ghz / cpu_ghz)))
                .normalize()
                .quantile(process.min_reliability)
            )

            target_delay = process.max_delay_ms - avg_network_delay
            min_cpu_ghz = exponential_search(gamma_func, target_delay, tolerance=0.1)

            self._process_CPU_demand[process_name] = min_cpu_ghz

        cpu_demand = self._process_CPU_demand[process_name]
        debug(f"Process {process_name} requires {cpu_demand:.2f} GHz")

        # iterate all hosts, and find the one with minimum cost
        host_costs = {}
        for host_name, host in self.context.hosts.items():
            # Check if the host can host the process
            cpu_utilized = cpu_demand + sum(v["allocated_cpu"] for v in self._allocation_map.get(host_name, []))
            if cpu_utilized > host.cpu_ghz:
                debug(f"  Host {host_name} cannot host process {process_name} (not enough CPU)")
                continue
            allocation_cost = 0
            # E_O = CPU usage on the host + CPU usage on this host
            # But E_O is the same for all hosts, since we are adding the same CPU demand to all hosts
            # Hence, we can ignore the CPU usage on this host
            # E_Q = network delay between process and host
            E_Q = self.context.links["gamma_com"][(process_name, host_name)]
            allocation_cost = E_Q
            host_costs[host_name] = allocation_cost
            debug(f"  Host {host_name} cost: E_Q={E_Q:.2f} = {allocation_cost:.2f}")
        # select the host with minimum cost (if any)
        selected_host = min(host_costs, key=host_costs.get, default=None)
        if selected_host is None:
            info(f"No host can host process {process_name} (not enough CPU)")
            return
        info(f"Selected host for process {process_name}: {selected_host} with cost {host_costs[selected_host]:.2f}")

        # Compute the CPU share for the process on the selected host
        cpu_share = cpu_demand / self.context.hosts[selected_host].cpu_ghz
        debug(f"CPU share for process {process_name} on host {selected_host}: {cpu_share*100:.2f}%")

        if selected_host not in self._allocation_map:
            self._allocation_map[selected_host] = []
        self._allocation_map[selected_host].append(
            {"process_name": process_name, "num_mns": 1, "cpu_share": cpu_share, "allocated_cpu": cpu_demand}
        )
        debug(f"Allocation map: {self._allocation_map}")
        debug("-----")


if __name__ == "__main__":
    moera = MOERA()
    moera.set_context("configs/scenario1.json")
    moera.add_1_mn("P0")
    moera.add_1_mn("P0")
    moera.add_1_mn("P1")
