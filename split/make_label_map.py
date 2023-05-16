import json

import pandas as pd

input = "ucfTrainTestlist/classInd.txt"
output = "ucfTrainTestlist/label-map.json"
df = pd.read_csv(
    input,
    delimiter=" ",
    header=None,
    names=["idx", "name"],
    index_col="idx",
    engine="c"
)

data = df["name"].to_dict()

with open(output, "w") as f:
    json.dump(data, f)
