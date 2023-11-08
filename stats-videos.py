from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

dataset_path = Path("/nas.dbms/randy/datasets/ucf101")
cols = "filename", "width", "height", "n_frames", "fps"
stats = []
extension = ".avi"

assert dataset_path.exists() and dataset_path.is_dir()

n_videos = len([f for f in dataset_path.rglob(f"**/*{extension}")])

with tqdm(total=n_videos) as bar:
    for file in dataset_path.glob("**/*.avi"):
        cap = cv2.VideoCapture(str(file))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        stats.append([file.name, width, height, n_frames, fps])
        bar.update(1)

desc = pd.DataFrame(stats, columns=cols).describe()
print(desc)
