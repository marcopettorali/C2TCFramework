import json

with open("results_scenario1_het1_13_0_TEST.json", "r") as f:
    results = json.load(f)

print(results)

data = {}
for rep_index, rep_data in results.items():
    for alg_name, alg_data in rep_data.items():
        if alg_name == "MOERA":
            continue
        if alg_name == "mns_arrival_list":
            continue

        _elem = {}

        for host_label, host_allocation in alg_data["allocation_map"].items():
            _elem[host_label] = []
            for split in host_allocation:
                num_mns = [x for x in alg_data["allocated_processes_mns"] if list(x.keys())[0] == split[0]][0][split[0]]
                _elem[host_label].append({"process_name": split[0].split("_")[0], "num_mns": num_mns, "cpu_share": split[1]})

        if alg_name not in data:
            data[alg_name] = []

        data[alg_name].append(_elem)

with open("results_scenario1_het1_13_0_TEST_converted.json", "w") as f:
    json.dump(data, f, indent=4)
