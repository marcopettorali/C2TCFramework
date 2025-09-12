import json

with open("results_scenario1_het1_13_0.json", "r") as f:
    results = json.load(f)

data = {}
for rep_name, rep_data in results.items():
    rep_name = int(rep_name)
   

    algorithms = list(rep_data.keys())
    for alg in algorithms:
        if alg not in data:
            data[alg] = []
        mns_allocated = sum(list(x.values())[0] for x in rep_data[alg]["allocated_processes_mns"])
        data[alg].append(mns_allocated)

from utils.logging import info
info(data)

# represent histograms of data for each algorithm in a single plot
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
for alg, values in data.items():
    plt.hist(values, bins=20, alpha=0.5, label=alg)
plt.xlabel("MNs allocated")
plt.ylabel("Frequency")
plt.title("Histogram of MNs allocated by Algorithm")
plt.legend()
plt.show()


# compute mean confidence intervals for each algorithm
from utils.stats import mean_confidence_interval, avg, ci_err
data = {alg: (avg(x := mean_confidence_interval(data[alg])), ci_err(x)) for alg in data}
print(data)
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()
algorithms = list(data.keys())
x = np.arange(len(algorithms))
means = [data[alg][0] for alg in algorithms]
errs = [data[alg][1] for alg in algorithms]
bars = ax.bar(x, means, yerr=errs, capsize=5)
ax.set_xticks(x)
ax.set_xticklabels(algorithms, rotation=45, ha="right")
ax.set_ylabel("Mean MNs allocated")
ax.set_title("Comparison of Algorithms")
plt.tight_layout()
plt.show()