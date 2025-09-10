from dataclasses import asdict
from resourceallocation.moera import MOERA
from resourceallocation.djnecora import DJNecora
import copy
import random
from utils.logging import focus, set_logging_level

SCENARIO = "scenario1"
NUM_REPETITIONS = 10  # 50
MAX_MNS = 13  # set to -1 to allocate all MNs of each process

context = DJNecora.load_context_from_file(f"configs/{SCENARIO}.json")

splitting_policies = ["no_splitting", "lazy_splitting", "greedy_splitting"]
selection_policies = ["first_fit", "next_fit", "best_fit", "worst_fit", "random_fit"]

results_file = f"out/djnecora_vs_moera_{SCENARIO}.json"


def _worker(rep_index):
    focus(f"--- REPETITION {rep_index + 1}/{NUM_REPETITIONS} ---")

    # Initialize algorithms
    moera = MOERA()
    moera.set_context(copy.deepcopy(context))

    djnecora_dict = {sp: {cp: DJNecora(sp, cp) for cp in selection_policies} for sp in splitting_policies}
    for sp in splitting_policies:
        for cp in selection_policies:
            djnecora_dict[sp][cp].set_context(copy.deepcopy(context))
            for process_name, process in context.processes.items():
                djnecora_dict[sp][cp].context.processes[process_name].mns = 1  # reset number of MNs to 1
            djnecora_dict[sp][cp].initialize_hosts()

    # get process list
    process_list = list(context.processes.keys())

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
                        focus(f"\t\tDJNecora ({sp}, {cp}): skipping allocation of MN {mn_index} of process {process_name} since previous MNs could not be allocated")

    # store results
    ret_dict = {}
    ret_dict["MOERA"] = moera._allocation_map
    for sp in splitting_policies:
        for cp in selection_policies:
            ret[f"DJ-NECORA.{sp}.{cp}"] = djnecora_dict[sp][cp]._allocation_table_per_host
    return ret


def run_experiments():

    # Run experiments

    results = {"MOERA": []}
    results.update({f"DJ-NECORA.{sp}.{cp}": [] for sp in splitting_policies for cp in selection_policies})

    import multiprocessing as mp

    # collect results from multiple processes
    with mp.Pool(processes=mp.cpu_count()) as pool:
        data = pool.map(_worker, range(NUM_REPETITIONS))

    focus("All repetitions completed")

    # process results
    for rep in data:
        for key, value in rep.items():
            results[key].append(value)

    focus("Results processed")

    # dump data to json
    import json

    with open(results_file, "w") as f:
        json.dump(results, f, indent=4, default=lambda o: asdict(o))


def plot_results():
    # load results from json
    import json

    with open(results_file, "r") as f:
        results = json.load(f)

    algorithms = list(results.keys())

    data = {alg: [] for alg in algorithms}

    for alg in algorithms:
        for rep_index in range(len(results[alg])):
            allocation = results[alg][rep_index]
            total_mns = 0
            for br in allocation:
                for split in range(len(allocation[br])):
                    total_mns += allocation[br][split]["num_mns"]
            data[alg].append(total_mns)

    # compute mean confidence intervals for each algorithm
    from utils.stats import mean_confidence_interval, avg, ci_err

    data = {alg: (avg(x := mean_confidence_interval(data[alg])), ci_err(x)) for alg in algorithms}

    def _get_color(alg_name):
        if alg_name == "MOERA":
            return "gray"
        elif "no_splitting" in alg_name:
            return "#80b1d3"
        elif "lazy_splitting" in alg_name:
            return "#b3de69"
        elif "greedy_splitting" in alg_name:
            return "#fb8072"
        else:
            raise ValueError(f"Unknown algorithm name {alg_name}")

    def _get_hatch(alg_name):
        if "first_fit" in alg_name:
            return ""
        elif "next_fit" in alg_name:
            return "//"
        elif "best_fit" in alg_name:
            return "xx"
        elif "worst_fit" in alg_name:
            return ".."
        elif "random_fit" in alg_name:
            return "oo"
        elif alg_name == "MOERA":
            return ""
        else:
            raise ValueError(f"Unknown algorithm name {alg_name}")

    def _normalized_label(label):
        split_map = {"no_splitting": "NS", "lazy_splitting": "LS", "greedy_splitting": "GS"}

        fit_map = {"first_fit": "FF", "next_fit": "NF", "best_fit": "BF", "worst_fit": "WF", "random_fit": "RF"}
        if not label.startswith("DJ-NECORA"):
            # se non è nel formato DJ-NECORA, ritorna invariato
            return label
        try:
            _, split_type, fit_type = label.split(".")
            return f"{split_map.get(split_type, split_type)}-{fit_map.get(fit_type, fit_type)}"
        except ValueError:
            # se il formato non è quello atteso, ritorna la label originale
            return label

    import matplotlib.pyplot as plt
    import matplotlib.transforms as mtransforms
    from utils.plotting import latex_initialize, bold

    latex_initialize()

    # bar plot
    fig, ax = plt.subplots()
    bar_width = 0.7
    x = range(len(algorithms))
    bars = []
    for i, alg in enumerate(algorithms):
        mean, err = data[alg]
        bars.append(
            ax.bar(
                i,
                mean,
                yerr=err,
                width=bar_width,
                label=_normalized_label(alg),
                color=_get_color(alg),
                hatch=_get_hatch(alg),
                capsize=5,
                edgecolor="black",
            )
        )

    ax.grid(axis="y")
    ax.set_axisbelow(True)

    ax.set_xlim(-0.5, len(algorithms) - 0.5)
    ax.set_xticks(x)
    labels = ax.set_xticklabels([bold(_normalized_label(alg)) for alg in algorithms])
    plt.setp(labels, rotation=30, horizontalalignment="right", fontsize=10)
    for label in labels:
        label.set_transform(label.get_transform() + mtransforms.ScaledTranslation(7 / 72, 2 / 72, ax.figure.dpi_scale_trans))

    ax.set_ylabel(bold("Allocated MNs"))

    plt.tight_layout()
    plt.show()


import os

# Check if results file does not exist
if not os.path.exists(results_file):
    run_experiments()

plot_results()
