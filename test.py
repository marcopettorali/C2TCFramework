ar1=[60, 60, 55, 58, 58, 56, 62, 62, 54, 53, 57, 59, 56, 56, 61, 56, 61, 54, 56, 61, 54, 60, 58, 61, 56, 54, 64, 54, 61, 61, 59, 53, 64, 69, 53, 56, 59, 54, 55, 69, 56, 58, 65, 61, 63, 62, 56, 60, 52, 57]
ar2=[62, 49, 48, 60, 43, 56, 64, 62, 58, 55, 51, 59, 48, 58, 43, 49, 46, 63, 46, 47, 56, 62, 51, 55, 47, 56, 60, 49, 46, 44, 61, 51, 48, 50, 55, 48, 61, 50, 57, 67, 52, 51, 44, 53, 65, 49, 50, 46, 47, 59]

ar_diff = [a - b for a, b in zip(ar1, ar2)]
print(ar_diff)

max_elem = max(ar_diff)
max_index = ar_diff.index(max_elem)
print(f"Max difference: {max_elem} at index {max_index}")

import json

with open("out/djnecora_initialfraction_scenario1_het1.json", "r") as f:
    data = json.load(f)

from utils.logging import print

print(data["0"]["mns_arrival_list"][max_index])
print(data["0"]["DJ-NECORA.no_splitting.worst_fit"][max_index])
print(data["0"]["DJ-NECORA.lazy_splitting.worst_fit"][max_index])