from resourceallocation.djnecora import DJNecora
import random
import itertools

from utils.logging import focus, info, set_logging_level

set_logging_level("focus")
SCENARIO = "scenario1"

context = DJNecora.load_context_from_file(f"configs/{SCENARIO}.json", _cpu_shares=[1 / i for i in range(1, 8 + 1)])

splitting_policies = ["no_splitting"]#, "lazy_splitting", "greedy_splitting"]
selection_policies = ["first_fit", "next_fit", "best_fit", "worst_fit", "random_fit"]

# shuffle process list
process_list = list(context.processes.keys())
random.shuffle(process_list)

for splitting_policy, selection_policy in itertools.product(splitting_policies, selection_policies):
    djnecora = DJNecora(splitting_policy, selection_policy)
    djnecora.set_context(context)
    djnecora.initialize_hosts()

    # allocate all processes with the required number of MNs
    for process in process_list:
        djnecora.add_process(process)

    # total number of MNs allocated

    total_mns = sum(p.num_mns for processes in djnecora._allocation_table_per_host.values() for p in processes)

    focus(splitting_policy, selection_policy, total_mns)
    focus(djnecora)