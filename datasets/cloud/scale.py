import json
from pathlib import Path

# load json in this folder "nach_20.json"

with open(Path(__file__).parent / "nach_20.json") as f:
    data = json.load(f)

    # scale all values
    data = [d/5 for d in data]

# save to new file
with open(Path(__file__).parent / "nach_20_x02.json", "w") as f:
    json.dump(data, f)
