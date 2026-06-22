# Conversion Standard

This page shows how to convert event sequence data from common formats into the Seahorse JSONL format.

## Target Format

One JSON object per line, one line per sequence:

```json
{"times": [0.1, 0.4, 1.2], "locations": [[0.2, 0.4], [0.3, 0.8], [0.7, 0.1]]}
```

## From a pandas DataFrame

If your data has one row per event with columns `seq_id`, `time`, `x`, `y`:

```python
import json
import pandas as pd

df = pd.read_csv("events.csv")  # columns: seq_id, time, x, y

with open("train.jsonl", "w") as f:
    for seq_id, group in df.groupby("seq_id"):
        group = group.sort_values("time")
        record = {
            "times": group["time"].tolist(),
            "locations": group[["x", "y"]].values.tolist(),
        }
        f.write(json.dumps(record) + "\n")
```

## From NumPy Arrays

If each sequence is stored as a NumPy array of shape `(N, 3)` — columns `(time, x, y)`:

```python
import json
import numpy as np

sequences = np.load("sequences.npy", allow_pickle=True)  # list of (N, 3) arrays

with open("train.jsonl", "w") as f:
    for seq in sequences:
        seq = seq[seq[:, 0].argsort()]  # sort by time
        record = {
            "times": seq[:, 0].tolist(),
            "locations": seq[:, 1:].tolist(),
        }
        f.write(json.dumps(record) + "\n")
```

## From a Flat CSV With Sequence IDs

```python
import json
import pandas as pd

df = pd.read_csv("events.csv")
# Expects columns: sequence_id, time, longitude, latitude

with open("data.jsonl", "w") as f:
    for sid, grp in df.sort_values(["sequence_id", "time"]).groupby("sequence_id"):
        record = {
            "times": grp["time"].tolist(),
            "locations": grp[["longitude", "latitude"]].values.tolist(),
        }
        f.write(json.dumps(record) + "\n")
```

## Splitting Into Train / Val / Test

After converting all sequences to a single JSONL file, split them by index:

```python
import json
import random

with open("data.jsonl") as f:
    records = [json.loads(line) for line in f]

random.seed(42)
random.shuffle(records)

n = len(records)
n_train = int(0.75 * n)
n_val = int(0.10 * n)

splits = {
    "train": records[:n_train],
    "val": records[n_train:n_train + n_val],
    "test": records[n_train + n_val:],
}

for split, seqs in splits.items():
    with open(f"{split}.jsonl", "w") as f:
        for seq in seqs:
            f.write(json.dumps(seq) + "\n")
```

## Validation

After converting, load with `load_jsonl` and run the [preparation checklist](add-dataset.md#preparation-checklist) before fitting.

```python
from seahorse import load_jsonl

train = load_jsonl("train.jsonl")
print(f"{len(train)} sequences, first has {len(train[0]['times'])} events")
```
