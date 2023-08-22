import json
import pathlib

import numpy as np
import pandas as pd

for csv in pathlib.Path("matrix").iterdir():
    df = pd.read_csv(csv, index_col=0, engine="pyarrow").astype(float)
    names_output_dir = pathlib.Path("relevant-names") / csv.stem
    ids_output_dir = pathlib.Path("relevant-ids") / csv.stem

    names_output_dir.mkdir(parents=True, exist_ok=True)
    ids_output_dir.mkdir(parents=True, exist_ok=True)

    for n in range(1, 6):
        sorted_ids = []
        sorted_names = []

        for i, row in enumerate(df.itertuples(index=False)):
            action = df.index[i]
            top_ids = np.argsort(row)[::-1][:n]
            top_names = df.columns[top_ids].to_list()

            sorted_ids.append({action: top_ids.tolist()})
            sorted_names.append({action: top_names})

        with open(ids_output_dir / f"top-{n}.json", "w") as f:
            json.dump(sorted_ids, f, indent=2)

        with open(names_output_dir / f"top-{n}.json", "w") as f:
            json.dump(sorted_names, f, indent=2)

    for thres in [i * 0.1 for i in range(1, 10)]:
        filtered_names = []
        filtered_ids = []

        for i, row in enumerate(df.itertuples(index=False)):
            action = df.index[i]
            ids_above = [i for i, val in enumerate(row) if val > thres]
            names_above = [col for col, val in zip(df.columns, row) if val > thres]

            filtered_names.append({action: names_above})
            filtered_ids.append({action: ids_above})

        with open(ids_output_dir / f"{thres:.1}.json", "w") as f:
            json.dump(filtered_ids, f)

        with open(names_output_dir / f"{thres:.1}.json", "w") as f:
            json.dump(filtered_names, f, indent=2)
