from pathlib import Path

import cv2
from tqdm import tqdm

dataset_path = Path("/nas.dbms/randy/datasets/ucf101-split")
output_path = Path("/nas.dbms/randy/datasets/ucf101-frames")
n_videos = sum(1 for f in dataset_path.glob("**/*") if f.is_file())

with tqdm(total=n_videos) as bar:
    for split in dataset_path.iterdir():
        for action in split.iterdir():
            for video in action.iterdir():
                bar.set_description(video.name)

                cap = cv2.VideoCapture(str(video))
                frames_output_path = output_path / split.name / action.name / video.name
                frames_output_path.mkdir(parents=True, exist_ok=True)

                frame_count = 0

                while True:
                    ret, frame = cap.read()

                    if not ret:
                        break

                    frame_filename = frames_output_path / f"frame_{frame_count:04d}.jpg"
                    frame_count += 1

                    cv2.imwrite(str(frame_filename), frame)

                cap.release()
                bar.update(1)
