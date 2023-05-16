from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

dataset_path = Path("../ucf101")
n_videos = len([f for f in dataset_path.rglob("*") if f.is_file()])
stats = []
cols = "filename", "width", "height", "n_frames", "fps"

with tqdm(total=n_videos) as bar:
    for subdir in dataset_path.iterdir():
        for file in subdir.iterdir():
            cap = cv2.VideoCapture(str(file.absolute()))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            stats.append([file.name, width, height, n_frames, fps])
            bar.update(1)

pd.DataFrame(stats, columns=cols).to_csv("stats.csv", index=False)