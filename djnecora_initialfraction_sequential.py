from dataclasses import asdict
from resourceallocation.djnecora import DJNecora
import copy
import random
from resourceallocation.jnecora import JNecora
from utils.logging import focus, info, set_logging_level

SCENARIO = "scenario1_het1"
NUM_REPETITIONS = 100
MAX_MNS = 13  # -1  # set to -1 to allocate all MNs of each process
LOAD_MNS_LIST_FROM_FILE = None  # "results_scenario1_het1_13_0_TEST.json"

initial_fractions = [0, 0.5, 1]
splitting_policies = ["no_splitting", "lazy_splitting", "greedy_splitting"]
selection_policies = ["first_fit", "next_fit", "best_fit", "worst_fit", "random_fit"]

results_file = f"out/djnecora_initialfraction_{SCENARIO}.json"

JNECORA_RESULT = 67


def min_cpu_share_for_preallocation(context, process_name, host_label, num_mns):
    return min(
        (
            (p, h, c, m)
            for (p, h, c, m), g in context.links["gamma_tot_precomputed"].items()
            if p == process_name and h == host_label and g <= context.processes[process_name].max_delay_ms and m == num_mns
        ),
        key=lambda x: x[2],  # max supported MNs, min CPU share
        default=None,
    )


def run_experiments():

    # Run J-NECORA to get preallocation map
    jnecora_context = JNecora.load_context_from_file(f"configs/{SCENARIO}.json")
    jnecora = JNecora()
    jnecora.set_context(jnecora_context)

    allocation, total_mns = jnecora.allocate_all_processes_optimal()
    ret = jnecora.compute_max_mns_min_cpushare_for_allocation(allocation)

    preallocation_map = {}
    for allocation_elem in ret:
        process_name, host_label, cpu_share, num_mns = allocation_elem
        preallocation_map[process_name] = {
            "host": host_label,
            "cpu_share": cpu_share,
            "num_mns": num_mns,
        }
    global JNECORA_RESULT
    JNECORA_RESULT = total_mns

    focus("=== J-NECORA RESULT ===")
    focus(preallocation_map)

    # Run DJ-NECORA experiments
    results = {}
    for initial_frac in initial_fractions:
        focus(f"=== INITIAL FRACTION {initial_frac} ===")

        # set grain compatible with the results in DJ-NECORA conference paper
        context = DJNecora.load_context_from_file(
            f"configs/{SCENARIO}.json", _cpu_shares={"cpu_share_precision": 0.01, "cpu_share_round_precision": 2}
        )

        # compute the number of MNs that will be allocated with this initial fraction
        preallocation_fraction_data = {
            process_name: (
                num_mns := round(initial_frac * preallocation_map[process_name]["num_mns"]),
                host := preallocation_map[process_name]["host"],
                min_cpu_share_for_preallocation(context, process_name, host, num_mns)[2] if num_mns > 0 else None,
            )
            for process_name in context.processes.keys()
        }

        focus(preallocation_fraction_data)

        results[initial_frac] = {f"DJ-NECORA.{sp}.{cp}": [] for sp in splitting_policies for cp in selection_policies}
        results[initial_frac].update({"mns_arrival_list": []})
        for rep in range(NUM_REPETITIONS):
            focus(f"--- REPETITION {rep + 1}/{NUM_REPETITIONS} ---")

            # Initialize algorithms and preallocate processes
            djnecora_dict = {sp: {cp: DJNecora(sp, cp) for cp in selection_policies} for sp in splitting_policies}
            for sp in splitting_policies:
                for cp in selection_policies:
                    djnecora_dict[sp][cp].set_context(copy.deepcopy(context))
                    # for process_name, process in context.processes.items():
                    #     djnecora_dict[sp][cp].context.processes[process_name].mns =
                    djnecora_dict[sp][cp].initialize_hosts()

                    for process_name, (num_mns, host, min_cpu_share) in preallocation_fraction_data.items():
                        if num_mns == 0:
                            continue
                        djnecora_dict[sp][cp]._apply_allocation(
                            process_name,
                            {"host_label": host, "supported_mns": num_mns, "cpu_share": min_cpu_share},
                            djnecora_dict[sp][cp].context.processes[process_name],
                        )
                        info(djnecora_dict[sp][cp]._allocation_table_per_host)
                        info(djnecora_dict[sp][cp]._available_resources_per_host)

            set_logging_level("focus")

            # Build MNs allocation list if not loading from file
            if LOAD_MNS_LIST_FROM_FILE is None:

                # get process list
                process_list = list(context.processes.keys())
                # get MNs list
                mns_list = [p for p in process_list for _ in range(MAX_MNS - preallocation_fraction_data[p][0])]

                # shuffle process list
                random.shuffle(mns_list)
            else:
                import json

                with open(LOAD_MNS_LIST_FROM_FILE, "r") as f:
                    loaded_track = json.load(f)

                mns_list = [x[0] for x in loaded_track[str(rep)]["mns_arrival_list"]]

            # transform each entry of the shuffled list in (process, i), where i is the index of the MN relative to process from 0 to N
            # e.g. [P0, P1, P0, P2, P1] -> [(P0,0), (P1,0), (P0,1), (P2,0), (P1,1)]
            mns_list = [(p, sum(1 for x in mns_list[:i] if x == p)) for i, p in enumerate(mns_list)]

            results[initial_frac]["mns_arrival_list"].append(mns_list)

            # Keep track of allocation status for DJNecora (stop allocating MNs of a process if one MN could not be allocated)
            allocation_status = {sp: {cp: {} for cp in selection_policies} for sp in splitting_policies}

            # Extract one element at a time and allocate it using all the algorithms
            for process_name, mn_index in mns_list:
                info(f"\tAllocating MN {mn_index} of process {process_name}")

                # DJNecora
                for sp in splitting_policies:
                    for cp in selection_policies:
                        # if first mn of the process, call add_process, else call add_1_mn
                        if mn_index == 0:
                            ret = djnecora_dict[sp][cp].add_process(process_name)
                            allocation_status[sp][cp][process_name] = ret
                            info(f"\t\tDJNecora ({sp}, {cp}): adding process {process_name}: **{'succeeded' if ret else 'failed'}**")
                        else:
                            if allocation_status[sp][cp][process_name]:
                                ret = djnecora_dict[sp][cp].add_1_mn_to_process(process_name)
                                allocation_status[sp][cp][process_name] = ret
                                info(
                                    f"\t\tDJNecora ({sp}, {cp}): adding 1 MN to process {process_name}: **{'succeeded' if ret else 'failed'}**"
                                )
                            else:
                                info(
                                    f"\t\tDJNecora ({sp}, {cp}): skipping allocation of MN {mn_index} of process {process_name} since previous MNs could not be allocated"
                                )

            # store results
            for sp in splitting_policies:
                for cp in selection_policies:
                    results[initial_frac][f"DJ-NECORA.{sp}.{cp}"].append(djnecora_dict[sp][cp]._allocation_table_per_host)

            # dump data to json
            import json

            with open(results_file, "w") as f:
                json.dump(results, f, indent=4, default=lambda o: asdict(o))


def plot_results():
    context = DJNecora.load_context_from_file(f"configs/{SCENARIO}.json")
    applications = list(context.processes.keys())

    # load results from json
    import json
    from utils.stats import mean_confidence_interval, avg, ci_err

    with open(results_file, "r") as f:
        results = json.load(f)

    # 1 plot per splitting policy
    for sp in splitting_policies:

        data = {cp: [] for cp in selection_policies}
        for cp in selection_policies:
            for i in initial_fractions:
                key = f"DJ-NECORA.{sp}.{cp}"
                tot_mns_for_app = [
                    sum(x["num_mns"] for br, br_data in rep_data.items() for x in br_data) for rep_data in results[str(i)][key]
                ]
                data[cp].append((avg(ulb := mean_confidence_interval(tot_mns_for_app)), ci_err(ulb)))

        data = {"-".join([x.capitalize() for x in k.split("_")]): v for k, v in data.items()}
        data = {k.replace("Random-Fit", "Random"): v for k, v in data.items()}

        from utils.plotting import latex_initialize, bold, grouped_bar_plot
        import matplotlib.pyplot as plt

        latex_initialize()

        color = [{"no_splitting": "#80b1d3", "lazy_splitting": "#b3de69", "greedy_splitting": "#fb8072"}[sp] for _ in selection_policies]
        hatches = ["", "o", "x"]

        fig, ax = plt.subplots()
        grouped_bar_plot(
            fig,
            ax,
            data,
            column_labels=[bold(f"I={int(i*100)}\\%") for i in initial_fractions],
            group_by_rows=True,
            colors=color,
            hatches=hatches,
            edgecolor="black",
        )

        ax.axhline(y=JNECORA_RESULT, color="red", linestyle="--")

        ax.set_xlabel(bold("Host selection policy"))
        ax.set_ylabel(bold("Total MNs allocated"))
        ax.set_ylim(0, 90)
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        ax.legend(ncols=3)
        fig.tight_layout()
        fig.savefig(f"out/plots/djnecora_initialfraction_{SCENARIO}_{sp}.pdf")

    # DATA BY APPLICATION
    # 1 plot per splitting policy AND application
    import itertools

    for sp, p in itertools.product(splitting_policies, applications):

        data = {cp: [] for cp in selection_policies}
        for cp in selection_policies:
            for i in initial_fractions:
                key = f"DJ-NECORA.{sp}.{cp}"
                tot_mns_for_app = [
                    sum(x["num_mns"] for br, br_data in rep_data.items() for x in br_data if x["process_name"] == p)
                    for rep_data in results[str(i)][key]
                ]
                data[cp].append((avg(ulb := mean_confidence_interval(tot_mns_for_app)), ci_err(ulb)))

        data = {"-".join([x.capitalize() for x in k.split("_")]): v for k, v in data.items()}
        data = {k.replace("Random-Fit", "Random"): v for k, v in data.items()}

        from utils.plotting import latex_initialize, bold, grouped_bar_plot
        import matplotlib.pyplot as plt

        latex_initialize()

        color = [{"no_splitting": "#80b1d3", "lazy_splitting": "#b3de69", "greedy_splitting": "#fb8072"}[sp] for _ in selection_policies]
        hatches = ["", "o", "x"]

        fig, ax = plt.subplots()
        grouped_bar_plot(
            fig,
            ax,
            data,
            column_labels=[bold(f"I={int(i*100)}\\%") for i in initial_fractions],
            group_by_rows=True,
            colors=color,
            hatches=hatches,
            edgecolor="black",
        )

        # ax.axhline(y=jnecora_result, color="red", linestyle="--")

        ax.set_xlabel(bold("Host selection policy"))
        ax.set_ylabel(bold("Total MNs allocated"))
        ax.set_ylim(0, 16)
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        ax.legend(ncols=3)
        fig.tight_layout()
        fig.savefig(f"out/plots/djnecora_initialfraction_{SCENARIO}_{sp}_{p}.pdf")

    # DEBUG PLOTS: no splitting vs lazy splitting per app with worstfit and I = 0

    data = {}
    for p in applications:
        for cp in ["worst_fit"]:
            for i in ["0"]:
                for sp in ["no_splitting", "lazy_splitting"]:
                    key = f"DJ-NECORA.{sp}.{cp}"
                    tot_mns_for_app = [
                        sum(x["num_mns"] for br, br_data in rep_data.items() for x in br_data if x["process_name"] == p)
                        for rep_data in results[str(i)][key]
                    ]
                    ci = (avg(ulb := mean_confidence_interval(tot_mns_for_app)), ci_err(ulb))

                    data.setdefault(p, []).append(ci)

    # plot grouped bar plot
    from utils.plotting import latex_initialize, bold, grouped_bar_plot
    import matplotlib.pyplot as plt
    latex_initialize()
    fig, ax = plt.subplots()
    grouped_bar_plot(
        fig,
        ax,
        data,
        column_labels=[bold("No Splitting"), bold("Lazy Splitting")],
        group_by_rows=True,
        colors=["#80b1d3", "#b3de69"],
        hatches=["", ""],
        edgecolor="black",
    )
    ax.set_xlabel(bold("Application"))
    ax.set_ylabel(bold("Total MNs allocated"))
    ax.set_ylim(0, 16)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.legend(ncols=3)
    fig.tight_layout()
    fig.savefig(f"out/plots/DEBUG_djnecora_initialfraction_{SCENARIO}_no_vs_lazy_worstfit_app.pdf")

import os

# Check if results file does not exist
if not os.path.exists(results_file):
    run_experiments()

plot_results()
