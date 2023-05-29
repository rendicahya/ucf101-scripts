from pathlib import Path

import cv2
from utils import iterate

input_path = Path("/nas.dbms/randy/datasets/ucf101")
output_path = Path("/nas.dbms/randy/datasets/ucf101-frames")


def split(action: Path, video: Path):
    cap = cv2.VideoCapture(str(video))
    frames_output_path = output_path / action.name / video.name
    frame_count = 0

    frames_output_path.mkdir(parents=True, exist_ok=True)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_filename = frames_output_path / f"frame_{frame_count:04d}.jpg"
        frame_count += 1

        cv2.imwrite(str(frame_filename), frame)

    cap.release()


iterate(input_path, split)
