from dataclasses import asdict
from resourceallocation.jnecora import JNecora
from resourceallocation.moera import MOERA, MOERAWrapper
from resourceallocation.djnecora import DJNecora
from resourceallocation.ojstr import OJSTRWrapper
import copy
import random
from utils.logging import focus, info, set_logging_level

SCENARIO = "scenario1_het1"
NUM_REPETITIONS = 100
MAX_MNS = 13  # -1  # set to -1 to allocate all MNs of each process
TRACK_FILE = "out/djnecora_initialfraction_scenario1_het1.json"  # None

# set coarse grain
context = JNecora.load_context_from_file(
    f"configs/{SCENARIO}.json", cpu_shares_descriptor={"cpu_share_precision": 0.01, "cpu_share_round_precision": 2}
)

results_file = f"out/djnecora_vs_moera_vs_ojstr_{SCENARIO}.json"
set_logging_level("focus")


def run_experiments():

    # Run experiments

    results = {"MOERA merge false": [], "MOERA merge true": [], "OJSTR merge false": [], "OJSTR merge true": []}

    for rep in range(NUM_REPETITIONS):
        focus(f"--- REPETITION {rep + 1}/{NUM_REPETITIONS} ---")

        # Initialize algorithms
        moera_merge_false = MOERAWrapper(copy.deepcopy(context), merge_vms=False)
        moera_merge_true = MOERAWrapper(copy.deepcopy(context), merge_vms=True)

        ojstr_merge_false = OJSTRWrapper(copy.deepcopy(context), merge_vms=False)
        ojstr_merge_true = OJSTRWrapper(copy.deepcopy(context), merge_vms=True)

        set_logging_level("focus")

        import json

        with open(TRACK_FILE, "r") as f:
            loaded_track = json.load(f)

        mns_list = [x[0] for x in loaded_track["0"]["mns_arrival_list"][rep]]

        # transform each entry of the shuffled list in (process, i), where i is the index of the MN relative to process from 0 to N
        # e.g. [P0, P1, P0, P2, P1] -> [(P0,0), (P1,0), (P0,1), (P2,0), (P1,1)]
        mns_list = [(p, sum(1 for x in mns_list[:i] if x == p)) for i, p in enumerate(mns_list)]

        # Extract one element at a time and allocate it using all the algorithms
        for process_name, mn_index in mns_list:
            info(f"\tAllocating MN {mn_index} of process {process_name}")

            # MOERA
            moera_merge_false.add_1_mn(process_name)
            moera_merge_true.add_1_mn(process_name)

            # OJSTR
            ojstr_merge_false.add_1_mn(process_name)
            ojstr_merge_true.add_1_mn(process_name)

        # finalize allocation
        moera_plan_merge_false = moera_merge_false.get_plan()
        moera_plan_merge_true = moera_merge_true.get_plan()

        ojstr_plan_merge_false = ojstr_merge_false.compute_final_allocation()
        ojstr_plan_merge_true = ojstr_merge_true.compute_final_allocation()

        # store results
        results["MOERA merge false"].append(moera_plan_merge_false)
        results["MOERA merge true"].append(moera_plan_merge_true)
        results["OJSTR merge false"].append(ojstr_plan_merge_false)
        results["OJSTR merge true"].append(ojstr_plan_merge_true)

        # dump data to json
        import json

        with open(results_file, "w") as f:
            json.dump(results, f, indent=4, default=lambda o: asdict(o))


# MATERIAL TO BE USED TO COMPUTE SUPPORTED MNs
test_context = JNecora.load_context_from_file(
    f"configs/{SCENARIO}.json", cpu_shares_descriptor={"cpu_share_precision": 0.01, "cpu_share_round_precision": 2}
)


def compute_supported(host_label, split_allocation):
    # floor cpu share to match test_context precision
    if test_context.hosts[host_label].infinite_parallelism:
        cpu_share = 1.0
    else:
        cpu_share = int(split_allocation["cpu_share"] * 100) / 100.0
    max_delay_ms = test_context.processes[split_allocation["process_name"]].max_delay_ms

    supported = 0
    for num_mns in range(1, split_allocation["num_mns"] + 1):
        gamma_tot = test_context.links["gamma_tot_precomputed"][(split_allocation["process_name"], host_label, cpu_share, num_mns)]
        if gamma_tot <= max_delay_ms:
            supported += 1
        else:
            break
    return supported


# ################################


def plot_results():

    set_logging_level("info")
    # load results from json
    import json
    from utils.stats import mean_confidence_interval, avg, ci_err

    # ORACLE (run now)
    from resourceallocation.oracle import Oracle

    oracle = Oracle()
    oracle.set_context(context)
    oracle_allocation = oracle.allocate_all_processes(max_mns_per_process=13, allocation_mode="max_apps_max_mns", splitting_mode="optimal")
    total_mns_oracle = sum(m for host_alloc in oracle_allocation.values() for _, m, _ in host_alloc)
    focus(oracle_allocation)

    # MOERA and OJSTR
    with open(results_file, "r") as f:
        results = json.load(f)

    # DJ-NECORA
    with open(TRACK_FILE, "r") as f:
        djnecora_results = json.load(f)

    splitting_policies = ["lazy_splitting"]  # ["no_splitting", "lazy_splitting", "greedy_splitting"]
    selection_policies = ["worst_fit"]  # ["first_fit", "next_fit", "best_fit", "worst_fit", "random_fit"]

    # 1 plot per splitting policy
    for sp in splitting_policies:

        data = {}
        # ORACLE result
        data["Oracle"] = (total_mns_oracle, 0)

        # DJ-NECORA results
        data.update({cp: [] for cp in selection_policies})
        for cp in selection_policies:
            key = f"DJ-NECORA.{sp}.{cp}"
            tot_mns = [sum(x["num_mns"] for br, br_data in rep_data.items() for x in br_data) for rep_data in djnecora_results["0"][key]]
            data[cp] = avg(ulb := mean_confidence_interval(tot_mns)), ci_err(ulb)

        data = {"-".join([x.capitalize() for x in k.split("_")]): v for k, v in data.items()}
        data = {k.replace("Random-Fit", "Random"): v for k, v in data.items()}

        # MOERA and OJSTR results
        for algorithm_name, algorithm_results in results.items():
            data[f"{algorithm_name}(allocated, supported)"] = (
                (
                    avg(
                        ulb := mean_confidence_interval(
                            [sum(x["num_mns"] for br, br_data in rep_data.items() for x in br_data) for rep_data in algorithm_results]
                        )
                    ),
                    ci_err(ulb),
                ),
                (
                    avg(
                        ulb := mean_confidence_interval(
                            [
                                sum(compute_supported(br, x) for br, br_data in rep_data.items() for x in br_data)
                                for rep_data in algorithm_results
                            ]
                        )
                    ),
                    ci_err(ulb),
                ),
            )

        info(data)

        from utils.plotting import latex_initialize, bold, grouped_bar_plot
        import matplotlib.pyplot as plt

        latex_initialize()

        colors = [
            {"no_splitting": "#80b1d3", "lazy_splitting": "#b3de69", "greedy_splitting": "#fb8072"}[sp] for _ in selection_policies
        ] + ["#bebada", "#bebada", "#8dd3c7", "#8dd3c7", "#bebada", "#bebada", "#8dd3c7", "#8dd3c7", "#ffffb3"]

        fig, ax = plt.subplots()
        for i, alg in enumerate(data.keys()):
            ax.bar(
                i,
                data[alg][0],
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
        ax.set_ylim(0, 95)
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        fig.tight_layout()
        fig.savefig(f"out/plots/djnecora_comparison_{SCENARIO}_{sp}.pdf")


def debug_plots():
    import matplotlib.pyplot as plt
    import numpy as np
    import json
    from utils.stats import avg, mean_confidence_interval, ci_err
    from utils.plotting import latex_initialize, bold, grouped_bar_plot


    considered_hosts = [f"BR{i}" for i in range(0, 5 + 1)] + ["CN"]  # TODO CHECK
    considered_apps = [f"P{i}" for i in range(0, 7 + 1)]  # TODO CHECK

    def map_results_key(key: str) -> str:
        return key.replace("merge true", "merge").replace("merge false", "no merge")


    # MOERA and OJSTR
    with open(results_file, "r") as f:
        moera_ojstr_results = json.load(f)

    # DJ-NECORA
    with open(TRACK_FILE, "r") as f:
        djnecora_results = json.load(f)["0"]["DJ-NECORA.lazy_splitting.worst_fit"]

    # build combined results dictionary
    allocated_results = {}
    allocated_results.update(moera_ojstr_results)
    allocated_results["DJ-NECORA"] = djnecora_results

    # build supported dictionary
    supported_results = copy.deepcopy(allocated_results)
    for algorithm_name, allocations_by_rep in supported_results.items():
        for i, allocation in enumerate(allocations_by_rep):
            for host_label, host_alloc in allocation.items():
                for j, split_allocation in enumerate(host_alloc):
                    copied_split_allocation = copy.deepcopy(split_allocation)
                    copied_split_allocation["num_mns"] = compute_supported(host_label, split_allocation)
                    supported_results[algorithm_name][i][host_label][j] = copied_split_allocation
    
    for metric in ["allocated", "supported"]:
        results = allocated_results if metric == "allocated" else supported_results

        # plot avg number of splits per host and by app
        avg_splits_per_host = {}
        avg_splits_per_app = {}

        avg_mns_per_app = {}
        avg_mns_per_host = {}
        for algorithm_name, allocations_by_rep in results.items():
            avg_splits_per_host[algorithm_name] = {}
            avg_splits_per_app[algorithm_name] = {}
            avg_mns_per_app[algorithm_name] = {}
            avg_mns_per_host[algorithm_name] = {}

            for allocation in allocations_by_rep:
                # count splits per host
                for host_name in considered_hosts:
                    allocation_on_host = allocation.get(host_name, [])

                    if host_name not in avg_splits_per_host[algorithm_name]:
                        avg_splits_per_host[algorithm_name][host_name] = []
                    avg_splits_per_host[algorithm_name][host_name].append(len(allocation_on_host))
                    
                    if host_name not in avg_mns_per_host[algorithm_name]:
                        avg_mns_per_host[algorithm_name][host_name] = []
                    num_mns_on_host = sum(x["num_mns"] for x in allocation_on_host)
                    avg_mns_per_host[algorithm_name][host_name].append(num_mns_on_host)

                # count splits per app
                for app_name in considered_apps:
                    if app_name not in avg_splits_per_app[algorithm_name]:
                        avg_splits_per_app[algorithm_name][app_name] = []
                    num_splits_for_app = sum(1 for host_alloc in allocation.values() for x in host_alloc if x["process_name"] == app_name)
                    avg_splits_per_app[algorithm_name][app_name].append(num_splits_for_app)

                    if app_name not in avg_mns_per_app[algorithm_name]:
                        avg_mns_per_app[algorithm_name][app_name] = []
                    num_mns_for_app = sum(
                        x["num_mns"] for host_alloc in allocation.values() for x in host_alloc if x["process_name"] == app_name
                    )
                    avg_mns_per_app[algorithm_name][app_name].append(num_mns_for_app)

            for host_name in avg_splits_per_host[algorithm_name]:
                avg_splits_per_host[algorithm_name][host_name] = (
                    avg(ulb := mean_confidence_interval(avg_splits_per_host[algorithm_name][host_name])),
                    ci_err(ulb),
                )

            for app_name in avg_splits_per_app[algorithm_name]:
                avg_splits_per_app[algorithm_name][app_name] = (
                    avg(ulb := mean_confidence_interval(avg_splits_per_app[algorithm_name][app_name])),
                    ci_err(ulb),
                )

            for app_name in avg_mns_per_app[algorithm_name]:
                avg_mns_per_app[algorithm_name][app_name] = (
                    avg(ulb := mean_confidence_interval(avg_mns_per_app[algorithm_name][app_name])),
                    ci_err(ulb),
                )

            for host_name in avg_mns_per_host[algorithm_name]:
                avg_mns_per_host[algorithm_name][host_name] = (
                    avg(ulb := mean_confidence_interval(avg_mns_per_host[algorithm_name][host_name])),
                    ci_err(ulb),
                )

        latex_initialize()

        # PLOTS PER HOST
        data_matrix_per_host = {}
        for algorithm_name, splits_per_host in avg_splits_per_host.items():
            data_matrix_per_host[map_results_key(algorithm_name)] = [splits_per_host.get(host, (0, 0)) for host in considered_hosts]

        for group_by_columns in [True, False]:
            fig, ax = plt.subplots(figsize=(10, 6))
            grouped_bar_plot(
                fig,
                ax,
                data_matrix_per_host,
                column_labels=[bold(h) for h in considered_hosts],
                colors=["#80b1d3", "#b3de69", "#fb8072", "#bebada", "#8dd3c7", "#ffffb3", "#fccde5", "#d9d9d9"],
                group_by_columns=group_by_columns,
            )

            ax.grid(axis="y")
            ax.set_axisbelow(True)
            ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

            ax.set_ylabel(bold("Avg number of splits per host"))
            ax.set_xlabel(bold("Host"))
            fig.legend(
                loc="upper center",
                ncol=3,
            )
            fig.tight_layout()
            fig.savefig(f"out/plots/DEBUG_djnecora_comparison_{SCENARIO}_splits_per_host_by_{'alg' if group_by_columns else 'host'}_{metric}.pdf")

        # PLOTS PER APP

        data_matrix_per_app = {}
        for algorithm_name, splits_per_app in avg_splits_per_app.items():
            data_matrix_per_app[map_results_key(algorithm_name)] = [splits_per_app.get(app, (0, 0)) for app in considered_apps]

        for group_by_columns in [True, False]:
            fig, ax = plt.subplots(figsize=(10, 6))
            grouped_bar_plot(
                fig,
                ax,
                data_matrix_per_app,
                column_labels=[bold(a) for a in considered_apps],
                colors=["#80b1d3", "#b3de69", "#fb8072", "#bebada", "#8dd3c7", "#ffffb3", "#fccde5", "#d9d9d9"],
                group_by_columns=group_by_columns,
            )

            ax.grid(axis="y")
            ax.set_axisbelow(True)
            ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
            ax.set_ylabel(bold("Avg number of splits per app"))
            ax.set_xlabel(bold("Application"))
            fig.legend(
                loc="upper center",
                ncol=3,
            )
            fig.tight_layout()
            fig.savefig(f"out/plots/DEBUG_djnecora_comparison_{SCENARIO}_splits_per_app_by_{'alg' if group_by_columns else 'app'}_{metric}.pdf")

        # PLOTS AVG MNS PER APP
        data_matrix_mns_per_app = {}
        for algorithm_name, mns_per_app in avg_mns_per_app.items():
            data_matrix_mns_per_app[map_results_key(algorithm_name)] = [mns_per_app.get(app, (0, 0)) for app in considered_apps]

        for group_by_columns in [True, False]:
            fig, ax = plt.subplots(figsize=(10, 6))
            grouped_bar_plot(
                fig,
                ax,
                data_matrix_mns_per_app,
                column_labels=[bold(a) for a in considered_apps],
                colors=["#80b1d3", "#b3de69", "#fb8072", "#bebada", "#8dd3c7", "#ffffb3", "#fccde5", "#d9d9d9"],
                group_by_columns=group_by_columns,
            )

            ax.grid(axis="y")
            ax.set_axisbelow(True)
            ax.set_ylim(0,14)
            ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
            ax.set_ylabel(bold("Avg number of MNs per app"))
            ax.set_xlabel(bold("Application"))
            fig.legend(
                loc="upper center",
                ncol=3,
            )
            fig.tight_layout()
            fig.savefig(f"out/plots/DEBUG_djnecora_comparison_{SCENARIO}_mns_per_app_by_{'alg' if group_by_columns else 'app'}_{metric}.pdf")

        # PLOTS AVG MNS PER HOST
        data_matrix_mns_per_host = {}
        for algorithm_name, mns_per_host in avg_mns_per_host.items():
            data_matrix_mns_per_host[map_results_key(algorithm_name)] = [mns_per_host.get(host, (0, 0)) for host in considered_hosts]
        
        for group_by_columns in [True, False]:  
            fig, ax = plt.subplots(figsize=(10, 6))
            grouped_bar_plot(
                fig,
                ax,
                data_matrix_mns_per_host,
                column_labels=[bold(h) for h in considered_hosts],
                colors=["#80b1d3", "#b3de69", "#fb8072", "#bebada", "#8dd3c7", "#ffffb3", "#fccde5", "#d9d9d9"],
                group_by_columns=group_by_columns,
            )

            ax.grid(axis="y")
            ax.set_axisbelow(True)
            ax.set_ylim(0,27)
            ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
            ax.set_ylabel(bold("Avg number of MNs per host"))
            ax.set_xlabel(bold("Host"))
            fig.legend(
                loc="upper center",
                ncol=3,
            )
            fig.tight_layout()
            fig.savefig(f"out/plots/DEBUG_djnecora_comparison_{SCENARIO}_mns_per_host_by_{'alg' if group_by_columns else 'host'}_{metric}.pdf")

import os

# # Check if results file does not exist
# if not os.path.exists(results_file):
#     run_experiments()

plot_results()
debug_plots()
