from dataclasses import asdict
from resourceallocation.moera import MOERA
from resourceallocation.djnecora import DJNecora
import copy
import random
from utils.logging import focus, set_logging_level

NUM_REPETITIONS = 30
MAX_MNS = 13  # set to -1 to allocate all MNs of each process

context = DJNecora.load_context_from_file("configs/scenario1.json")

splitting_policies = ["no_splitting", "lazy_splitting", "greedy_splitting"]
selection_policies = ["first_fit", "next_fit", "best_fit", "worst_fit", "random_fit"]

# Initialize algorithms
moera = MOERA()
moera.set_context(copy.deepcopy(context))

djnecora_dict = {sp: {cp: DJNecora(sp, cp) for cp in selection_policies} for sp in splitting_policies}
for sp in splitting_policies:
    for cp in selection_policies:
        djnecora_dict[sp][cp].set_context(copy.deepcopy(context))
        for process_name, process in context.processes.items():
            djnecora_dict[sp][cp].context.processes[process_name].num_mns = 1  # reset number of MNs to 1
        djnecora_dict[sp][cp].initialize_hosts()


set_logging_level("focus")

# Run experiments

results = {"MOERA": []}
results.update({f"DJ-NECORA.{sp}.{cp}": [] for sp in splitting_policies for cp in selection_policies})

for rep in range(NUM_REPETITIONS):
    focus(f"--- REPETITION {rep + 1}/{NUM_REPETITIONS} ---")

    # get process list
    process_list = list(moera.context.processes.keys())

    # get MNs list
    mns_list = [p for p in process_list for _ in range(0, MAX_MNS if MAX_MNS >= 0 else context.processes[p].mns)]

    # shuffle process list
    random.shuffle(mns_list)

    # transform each entry of the shuffled list in (process, i), where i is the index of the MN relative to process from 0 to N
    # e.g. [P0, P1, P0, P2, P1] -> [(P0,0), (P1,0), (P0,1), (P2,0), (P1,1)]
    mns_list = [(p, sum(1 for x in mns_list[:i] if x == p)) for i, p in enumerate(mns_list)]

    # Keep track of allocation status for DJNecora (stop allocating MNs of a process if one MN could not be allocated)
    allocation_status = {sp: {cp: {} for cp in selection_policies} for sp in splitting_policies}

    # Extract one element at a time and allocate it using all the algorithms
    for process_name, mn_index in mns_list:
        focus(f"\tAllocating MN {mn_index} of process {process_name}")

        # MOERA
        moera.add_1_mn(process_name)

        # DJNecora
        for sp in splitting_policies:
            for cp in selection_policies:
                # if first mn of the process, call add_process, else call add_1_mn
                if mn_index == 0:
                    ret = djnecora_dict[sp][cp].add_process(process_name)
                    allocation_status[sp][cp][process_name] = ret
                    focus(f"\t\tDJNecora ({sp}, {cp}): adding process {process_name}: **{'succeeded' if ret else 'failed'}**")
                else:
                    if allocation_status[sp][cp][process_name]:
                        ret = djnecora_dict[sp][cp].add_1_mn_to_process(process_name)
                        allocation_status[sp][cp][process_name] = ret
                        focus(f"\t\tDJNecora ({sp}, {cp}): adding 1 MN to process {process_name}: **{'succeeded' if ret else 'failed'}**")
                    else:
                        focus(
                            f"\t\tDJNecora ({sp}, {cp}): skipping allocation of MN {mn_index} of process {process_name} since previous MNs could not be allocated"
                        )

    # store results
    results["MOERA"].append(moera._allocation_map)
    for sp in splitting_policies:
        for cp in selection_policies:
            results[f"DJ-NECORA.{sp}.{cp}"].append(djnecora_dict[sp][cp]._allocation_table_per_host)

# dump data to json
import json

with open("out/djnecora_vs_moera_experiments.json", "w") as f:
    json.dump(results, f, indent=4, default=lambda o: asdict(o))
