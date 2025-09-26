import json
from utils.stats import mean_confidence_interval, avg, ci_err
from utils.logging import print

SCENARIOS = ["scenario1_het1", "scenario1_het2", "scenario1_het3", "scenario1_hom"]
ALGORITHM = "DJ-NECORA.lazy_splitting.first_fit"

# LOAD DATA
data_dict = {}
for scenario in SCENARIOS:
    with open(f"out/djnecora_initialfraction_{scenario}.json") as f:
        data = json.load(f)
    data_dict[scenario] = data

# ANALYZE DATA
results = {}
for scenario, scenario_data in data_dict.items():
    results[scenario] = []
    for initial_fraction, initial_fraction_data in scenario_data.items():
        mns_by_rep = []
        for rep_data in initial_fraction_data[ALGORITHM]:
            mns = sum(x["num_mns"] for br, br_allocation in rep_data.items() for x in br_allocation)
            mns_by_rep.append(mns)

        # compute CIs
        lo, hi = mean_confidence_interval(mns_by_rep)
        mean = avg([lo, hi])
        err = ci_err([lo, hi])
        results[scenario].append((mean, err))

print(results)

# PLOT RESULTS
from utils.plotting import latex_initialize, bold, grouped_bar_plot
import matplotlib.pyplot as plt

latex_initialize()

fig, ax = plt.subplots()
grouped_bar_plot(fig, ax, results, [0, 0.5, 1])
ax.set_xticklabels([bold(x.split("_")[-1].upper()) for x in SCENARIOS])
fig.tight_layout()
plt.show()
