import json
import pathlib

import pandas as pd

for csv in pathlib.Path("matrix").iterdir():
    df = pd.read_csv(csv, index_col=0, engine="pyarrow").astype(float)
    names_output_dir = pathlib.Path("relevant-names") / csv.stem
    ids_output_dir = pathlib.Path("relevant-ids") / csv.stem

    names_output_dir.mkdir(parents=True, exist_ok=True)
    ids_output_dir.mkdir(parents=True, exist_ok=True)

    for thres in [i * 0.1 for i in range(1, 10)]:
        filtered_names = []
        filtered_ids = []

        for i, row in enumerate(df.itertuples(index=False)):
            ids_above = [col for col, val in zip(df.columns, row) if val > thres]
            names_above = [col for col, val in zip(df.columns, row) if val > thres]
            action = df.index[i]

            filtered_names.append({action: names_above})
            filtered_ids.append({action: ids_above})

        with open(names_output_dir / f"{thres:.1}.json", "w") as f:
            json.dump(filtered_names, f, indent=2)

        with open(ids_output_dir / f"{thres:.1}.json", "w") as f:
            json.dump(filtered_ids, f)
