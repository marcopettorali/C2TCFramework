import json
from utils.stats import mean_confidence_interval, avg, ci_err
from utils.logging import print

SCENARIOS = ["scenario1_het1", "scenario1_het2", "scenario1_het3", "scenario1_hom"]
JNECORA_RESULTS = [67, 80, 80, 74]
ALGORITHM = "DJ-NECORA.lazy_splitting.worst_fit" #"DJ-NECORA.lazy_splitting.first_fit"

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
color = (lambda x: "#80b1d3" if "no" in x else "#b3de69" if "lazy" in x else "#fb8072")(ALGORITHM)

fig, ax = plt.subplots()
grouped_bar_plot(
    fig,
    ax,
    results,
    [bold(f"I={int(x*100)}\\%") for x in [0, 0.5, 1]],
    hatches=["", "o", "x"],
    colors=[color] * len(SCENARIOS),
    edgecolor="black",
    group_by_columns=True,
)
# Add JNECORA results

# horizontal lines for each JNECORA result
for i, scenario in enumerate(SCENARIOS):
    ax.hlines(
        JNECORA_RESULTS[i],
        i-0.15,
        i+0.7,
        colors="red",
        linestyles="dashed",
        label=None,
    )

ax.set_xticklabels([x.split("_")[-1].upper() for x in SCENARIOS])
ax.set_xlabel(bold("Scenario"))
ax.yaxis.grid(True)
ax.yaxis.set_major_locator(plt.MultipleLocator(10))
ax.set_ylim(0, 95)
ax.set_ylabel(bold("Total MNs Allocated"))
ax.set_axisbelow(True)

ax.legend(ncols=3, fontsize=16, loc="upper center")

fig.tight_layout()
fig.savefig(f"out/plots/djnecora_heterogeneous_plot_{ALGORITHM.replace('.', '_')}.pdf")
