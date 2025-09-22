from dataclasses import asdict
from resourceallocation.jnecora import JNecora
from resourceallocation.moera import MOERA
from resourceallocation.djnecora import DJNecora
import copy
import random
from utils.logging import focus, info, set_logging_level

SCENARIO = "scenario1_het1"
NUM_REPETITIONS = 50
MAX_MNS = 13  # -1  # set to -1 to allocate all MNs of each process
LOAD_MNS_LIST_FROM_FILE = "out/djnecora_initialfraction_scenario1_het1.json"  # None

# set coarse grain
context = DJNecora.load_context_from_file(
    f"configs/{SCENARIO}.json", _cpu_shares={"cpu_share_precision": 0.01, "cpu_share_round_precision": 2}
)

results_file = f"out/djnecora_vs_moera_{SCENARIO}.json"


def run_experiments():

    # Run experiments

    results = {"MOERA": []}

    for rep in range(NUM_REPETITIONS):
        focus(f"--- REPETITION {rep + 1}/{NUM_REPETITIONS} ---")

        # Initialize algorithms
        moera = MOERA()
        moera.set_context(copy.deepcopy(context))

        set_logging_level("focus")

        # Build MNs allocation list if not loading from file
        if LOAD_MNS_LIST_FROM_FILE is None:

            # get process list
            process_list = list(moera.context.processes.keys())

            # get MNs list
            mns_list = [p for p in process_list for _ in range(0, MAX_MNS if MAX_MNS >= 0 else context.processes[p].mns)]

            # shuffle process list
            random.shuffle(mns_list)
        else:
            import json

            with open(LOAD_MNS_LIST_FROM_FILE, "r") as f:
                loaded_track = json.load(f)

            mns_list = [x[0] for x in loaded_track["0"]["mns_arrival_list"][rep]]

        # transform each entry of the shuffled list in (process, i), where i is the index of the MN relative to process from 0 to N
        # e.g. [P0, P1, P0, P2, P1] -> [(P0,0), (P1,0), (P0,1), (P2,0), (P1,1)]
        mns_list = [(p, sum(1 for x in mns_list[:i] if x == p)) for i, p in enumerate(mns_list)]

        # Extract one element at a time and allocate it using all the algorithms
        for process_name, mn_index in mns_list:
            info(f"\tAllocating MN {mn_index} of process {process_name}")

            # MOERA
            moera.add_1_mn(process_name)

        # store results
        results["MOERA"].append(moera._allocation_map)

        # dump data to json
        import json

        with open(results_file, "w") as f:
            json.dump(results, f, indent=4, default=lambda o: asdict(o))


def plot_results():
    # load results from json
    import json
    from utils.stats import mean_confidence_interval, avg, ci_err

    # ORACLE (run now)
    from resourceallocation.oracle import Oracle

    oracle = Oracle()
    oracle.set_context(context)
    oracle_allocation = oracle.allocate_all_processes(max_mns_per_process=13)
    total_mns_oracle = sum(m for host_alloc in oracle_allocation.values() for _, m, _ in host_alloc)

    # MOERA
    with open(results_file, "r") as f:
        moera_results = json.load(f)["MOERA"]

    # DJ-NECORA
    with open(f"out/djnecora_initialfraction_{SCENARIO}.json", "r") as f:
        djnecora_results = json.load(f)

    splitting_policies = ["no_splitting", "lazy_splitting", "greedy_splitting"]
    selection_policies = ["first_fit", "next_fit", "best_fit", "worst_fit", "random_fit"]

    # 1 plot per splitting policy
    for sp in splitting_policies:
        data = {cp: [] for cp in selection_policies}
        for cp in selection_policies:
            key = f"DJ-NECORA.{sp}.{cp}"
            tot_mns = [sum(x["num_mns"] for br, br_data in rep_data.items() for x in br_data) for rep_data in djnecora_results["0"][key]]
            data[cp] = avg(ulb := mean_confidence_interval(tot_mns)), ci_err(ulb)

        data = {"-".join([x.capitalize() for x in k.split("_")]): v for k, v in data.items()}
        data = {k.replace("Random-Fit", "Random"): v for k, v in data.items()}

        # MOERA result
        data["MOERA"] = (
            avg(
                ulb := mean_confidence_interval(
                    [sum(x["num_mns"] for br, br_data in rep_data.items() for x in br_data) for rep_data in moera_results]
                )
            ),
            ci_err(ulb),
        )

        # ORACLE result
        data["Oracle"] = (total_mns_oracle, 0)

        info(data)

        from utils.plotting import latex_initialize, bold, grouped_bar_plot
        import matplotlib.pyplot as plt

        latex_initialize()

        colors = [
            {"no_splitting": "#80b1d3", "lazy_splitting": "#b3de69", "greedy_splitting": "#fb8072"}[sp] for _ in selection_policies
        ] + ["#bebada", "#8dd3c7"]

        fig, ax = plt.subplots()
        for i, alg in enumerate(data.keys()):
            ax.bar(
                i, data[alg][0],
                yerr=data[alg][1],
                label=alg,
                color=colors[i],
                capsize=4,
                edgecolor="black",
            )
        ax.set_xticks(range(len(data)))
        ax.set_xticklabels([bold(x) for x in list(data.keys())])
        plt.setp(ax.get_xticklabels(), fontsize=10)
        ax.set_xlabel(bold("Algorithm"))
        ax.set_ylabel(bold("Total MNs allocated"))
        ax.set_ylim(0, 90)
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        fig.tight_layout()
        fig.savefig(f"out/plots/djnecora_comparison_{SCENARIO}_{sp}.pdf")


import os

# Check if results file does not exist
if not os.path.exists(results_file):
    run_experiments()

plot_results()
