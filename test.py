from resourceallocation.jnecora import JNecora
from resourceallocation.oracle import solve_mn_allocation
from utils.logging import set_logging_level, info
import itertools
from collections import defaultdict

context = JNecora.load_context_from_file("configs/scenario1.json")

MAX_MNS = 13

min_cpu_dict = {}

for p, h in itertools.product(context.processes, context.hosts):
    # Add dummy entry for m=0 (no MNs)
    min_cpu_dict[(p, h, 0)] = 0.0

    max_delay_ms = context.processes[p].max_delay_ms

    # For each p,h,m get min cpu
    _temp = defaultdict(list)
    for k, val in context.links["gamma_tot_precomputed"].items():
        if k[0] == p and k[1] == h and val <= max_delay_ms and k[3] <= MAX_MNS:
            _temp[k[3]].append(k[2])

    for k, vals in _temp.items():
        if vals:
            min_cpu_dict[(p, h, k)] = min(vals)

# prepare host capacities
host_capacities_perc = {host_label: float("inf") if host.infinite_parallelism else 1.0 for host_label, host in context.hosts.items()}
ret = solve_mn_allocation(min_cpu_dict=min_cpu_dict, host_capacities_perc=host_capacities_perc)
info(ret)
