from resourceallocation.djnecora import DJNecora
import random
import itertools
import json
import os

from resourceallocation.jnecora import JNecora
from utils.logging import focus, info, set_logging_level

SCENARIO = "scenario1_het1"
NUM_REPETITIONS = 100

splitting_policies = ["no_splitting"]
selection_policies = ["first_fit", "next_fit", "best_fit", "worst_fit", "random_fit"]

results_file = f"out/djnecora_vs_jnecora_{SCENARIO}.json"

# COMPUTE J-NECORA SOLUTION
jnecora_context = JNecora.load_context_from_file(f"configs/{SCENARIO}.json")
jnecora = JNecora()
jnecora.set_context(jnecora_context)

allocation, total_mns = jnecora.allocate_all_processes_optimal()
if allocation is None:
    allocation, total_mns = jnecora.allocate_all_processes_besteffort(method="ga")

ret = jnecora.compute_max_mns_min_cpushare_for_allocation(allocation)
max_mns_for_jnecora = {x[0]: x[3] for x in ret}

focus(max_mns_for_jnecora)


def run_experiments():
    set_logging_level("focus")

    # DJ-NECORA EXPERIMENTS
    focus(f"Starting DJNecora experiments for scenario {SCENARIO} with {NUM_REPETITIONS} repetitions")
    context = DJNecora.load_context_from_file(f"configs/{SCENARIO}.json", cpu_shares=[1 / i for i in range(1, 8 + 1)])

    results = {sp: {cp: [] for cp in selection_policies} for sp in splitting_policies}

    for _seed in range(NUM_REPETITIONS):
        focus(f"Experiment repetition with seed {_seed}")
        # shuffle process list
        process_list = list(context.processes.keys())
        random.shuffle(process_list)

        for splitting_policy, selection_policy in itertools.product(splitting_policies, selection_policies):
            djnecora = DJNecora(splitting_policy, selection_policy)
            djnecora.set_context(context)
            djnecora.initialize_hosts()

            for process, max_mns in max_mns_for_jnecora.items():
                djnecora.context.processes[process].mns = max_mns

            # allocate all processes with the required number of MNs
            for process in process_list:
                djnecora.add_process(process)

            # total number of MNs allocated
            total_mns = sum(p.num_mns for processes in djnecora._allocation_table_per_host.values() for p in processes)

            results[splitting_policy][selection_policy].append(total_mns)

        # dump to json
        with open(f"out/djnecora_vs_jnecora_{SCENARIO}.json", "w") as f:
            json.dump(results, f, indent=4)

    focus(f"Results saved to {results_file}")


def plot_results():
    with open(f"out/djnecora_vs_jnecora_{SCENARIO}.json", "r") as f:
        results = json.load(f)

    from utils.stats import mean_confidence_interval
    from utils.plotting import latex_initialize, bold
    import itertools

    latex_initialize()

    ci_data = {}
    for splitting_policy, selection_policy in itertools.product(splitting_policies, selection_policies):
        data = results[splitting_policy][selection_policy]
        lb, ub = mean_confidence_interval(data)
        ci_data[(splitting_policy, selection_policy)] = (lb, ub)
        focus(
            f"Splitting: {splitting_policy:<15} Selection: {selection_policy:<12} => Avg MNs: {sum(data)/len(data):<6.2f} (95% CI: {lb:<6.2f}, {ub:<6.2f})"
        )

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    bar_width = 0.4
    x = range(len(selection_policies))
    for i, splitting_policy in enumerate(splitting_policies):
        means = [sum(results[splitting_policy][sp]) / len(results[splitting_policy][sp]) for sp in selection_policies]
        cis = [mean_confidence_interval(results[splitting_policy][sp]) for sp in selection_policies]
        lb = [m - ci[0] for m, ci in zip(means, cis)]
        ub = [ci[1] - m for m, ci in zip(means, cis)]
        ax.bar([p + i * bar_width for p in x], means, bar_width, yerr=[lb, ub], capsize=5, color="#80b1d3", edgecolor="black")
    ax.set_xticks([p + bar_width * (len(splitting_policies) - 1) / 2 for p in x])
    ax.set_xticklabels(["-".join([x.capitalize() for x in sp.split("_")]).replace("Random-Fit", "Random") for sp in selection_policies])
    ax.set_ylabel(bold("Total MNs Allocated"))
    ax.set_xlabel(bold("Host selection policy"))

    ax.set_xlim(-0.5,4.5)

    ax.hlines(
        y=sum(max_mns_for_jnecora.values()),
        xmin=-0.5,
        xmax=len(selection_policies) - 0.5,
        colors="r",
        linestyles="--",
        label=bold("Max MNs J-NECORA"),
    )
    ax.legend()
    ax.yaxis.grid(True)
    ax.yaxis.set_major_locator(plt.MultipleLocator(10))
    ax.set_axisbelow(True)
    ax.set_ylim(0, 95)
    plt.tight_layout()
    plt.savefig(f"out/plots/djnecora_vs_jnecora_{SCENARIO}.pdf")


if not os.path.exists(results_file):
    run_experiments()

plot_results()
