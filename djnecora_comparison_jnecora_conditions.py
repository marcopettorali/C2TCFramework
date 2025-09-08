from resourceallocation.djnecora import DJNecora
import random
import itertools
import json

from utils.logging import focus, info, set_logging_level

SCENARIO = "scenario1"
NUM_REPETITIONS = 50

splitting_policies = ["no_splitting"]  
selection_policies = ["first_fit", "next_fit", "best_fit", "worst_fit", "random_fit"]

# check if results file already exists
import os

results_file = f"out/djnecora_experiments_{SCENARIO}.json"
if not os.path.exists(results_file):
    focus(f"Starting DJNecora experiments for scenario {SCENARIO} with {NUM_REPETITIONS} repetitions")
    context = DJNecora.load_context_from_file(f"configs/{SCENARIO}.json", _cpu_shares=[1 / i for i in range(1, 8 + 1)])
    set_logging_level("focus")

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

            # allocate all processes with the required number of MNs
            for process in process_list:
                djnecora.add_process(process)

            # total number of MNs allocated
            total_mns = sum(p.num_mns for processes in djnecora._allocation_table_per_host.values() for p in processes)

            results[splitting_policy][selection_policy].append(total_mns)

    # dump to json
    with open(f"out/djnecora_experiments_{SCENARIO}.json", "w") as f:
        json.dump(results, f, indent=4)

    focus(f"Results saved to {results_file}")
else:
    focus(f"Results file {results_file} already exists, skipping experiments")  
    with open(f"out/djnecora_experiments_{SCENARIO}.json", "r") as f:
        results = json.load(f)

from utils.stats import mean_confidence_interval
import itertools

ci_data = {}
for splitting_policy, selection_policy in itertools.product(splitting_policies, selection_policies):
    data = results[splitting_policy][selection_policy]
    lb, ub = mean_confidence_interval(data)
    ci_data[(splitting_policy, selection_policy)] = (lb, ub)
    focus(f"Splitting: {splitting_policy:<15} Selection: {selection_policy:<12} => Avg MNs: {sum(data)/len(data):<6.2f} (95% CI: {lb:<6.2f}, {ub:<6.2f})")

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.15
x = range(len(selection_policies))
for i, splitting_policy in enumerate(splitting_policies):
    means = [sum(results[splitting_policy][sp]) / len(results[splitting_policy][sp]) for sp in selection_policies]
    cis = [mean_confidence_interval(results[splitting_policy][sp]) for sp in selection_policies]
    lb = [m - ci[0] for m, ci in zip(means, cis)]
    ub = [ci[1] - m for m, ci in zip(means, cis)]
    ax.bar([p + i * bar_width for p in x], means, bar_width, yerr=[lb, ub], capsize=5, label=splitting_policy)
ax.set_xticks([p + bar_width * (len(splitting_policies) - 1) / 2 for p in x])
ax.set_xticklabels(selection_policies)
ax.set_ylabel("Average Number of MNs Allocated")
ax.set_title(f"DJNecora Performance in {SCENARIO} Scenario ({NUM_REPETITIONS} Repetitions)")
ax.legend(title="Splitting Policy")
plt.tight_layout()
plt.show()