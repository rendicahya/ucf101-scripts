import pandas as pd

df = pd.read_csv("stats.csv", engine="pyarrow")
cols = "filename", "width", "height", "n_frames", "fps"

print(df[cols].describe())