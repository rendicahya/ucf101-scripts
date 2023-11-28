from pathlib import Path

import cv2
import pandas as pd
from assertpy.assertpy import assert_that
from python_config import Config
from python_file import count_files
from python_video import video_info
from tqdm import tqdm

conf = Config("config.json")
dataset_path = Path(conf.ucf101.path)
stats = []

assert_that(dataset_path).is_directory().is_readable()
n_videos = count_files(dataset_path, ext=conf.ucf101.ext)
bar = tqdm(total=n_videos)

for file in dataset_path.glob(f"**/*{conf.ucf101.ext}"):
    info = video_info(file)
    width, height = info["width"], info["height"]
    n_frames = info["n_frames"]
    fps = info["fps"]

    stats.append([file.name, width, height, n_frames, fps])
    bar.update(1)

cols = "filename", "width", "height", "n_frames", "fps"
desc = pd.DataFrame(stats, columns=cols).describe()

bar.close()
print(desc)
