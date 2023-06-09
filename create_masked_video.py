from pathlib import Path

import cv2
import numpy as np
import utils
from bs4 import BeautifulSoup
from moviepy.editor import ImageSequenceClip, VideoFileClip

annotation_path = Path("/nas.dbms/randy/projects/ucf101-scripts/annotations")
input_path = Path("/nas.dbms/randy/datasets/ucf101")
output_path = Path("/nas.dbms/randy/datasets/ucf101-masked")
action_only = False
mask_actor = False

colors = (
    (0, 0, 255),
    (0, 255, 0),
    (255, 0, 0),
    (0, 255, 255),
    (255, 0, 255),
    (255, 255, 0),
    (255, 255, 255),
)


def operation(action, anno_file):
    if not anno_file.suffix == ".xgtf":
        return

    with open(anno_file) as f:
        try:
            soup = BeautifulSoup(f, features="xml")
        except:
            return

    data = soup.find("data")
    people_bbox = {}

    if data is None:
        return

    for sourcefile in data.find_all("sourcefile"):
        for person in sourcefile.find_all("object", {"name": "PERSON"}):
            person_id = int(person["id"])

            if person_id in people_bbox:
                continue

            person_locations = person.find("attribute", {"name": "Location"})
            person_bbox = {}
            person_action = person.find("data:bvalue", {"value": "true"})

            if not person_action:
                continue

            act_start, act_end = [int(i) for i in person_action["framespan"].split(":")]

            for bbox in person_locations.find_all("data:bbox"):
                start, end = [int(i) for i in bbox["framespan"].split(":")]

                if action_only:
                    if act_start <= start <= act_end or act_start <= end <= act_end:
                        start = max(start, act_start)
                        end = min(end, act_end)

                        for frame in range(start - 1, end):
                            bbox_data = {
                                frame: (
                                    bbox["x"],
                                    bbox["y"],
                                    bbox["width"],
                                    bbox["height"],
                                )
                            }

                            person_bbox.update(bbox_data)
                else:
                    for frame in range(start - 1, end):
                        bbox_data = {
                            frame: (
                                bbox["x"],
                                bbox["y"],
                                bbox["width"],
                                bbox["height"],
                            )
                        }

                        person_bbox.update(bbox_data)

            people_bbox.update({person_id: person_bbox})

    input_video_path = input_path / action.name / (anno_file.with_suffix(".avi").name)
    output_video_path = output_path / action.name / (anno_file.with_suffix(".mp4").name)

    output_video_path.parent.mkdir(parents=True, exist_ok=True)

    clip = VideoFileClip(str(input_video_path))
    frames = clip.iter_frames()
    output_frames = []

    for i, frame in enumerate(frames):
        fn = np.zeros if mask_actor else np.ones
        mask = fn(frame.shape[:2], dtype=frame.dtype)

        for person_id, person_bbox in people_bbox.items():
            if i not in person_bbox:
                continue

            bbox = person_bbox[i]
            x1, y1, w, h = [int(i) for i in bbox]
            x2 = x1 + w
            y2 = y1 + h
            mask[y1:y2, x1:x2] = 1 if mask_actor else 0

        frame = cv2.bitwise_and(frame, frame, mask=mask)

        output_frames.append(frame)

    if len(output_frames) > 0:
        clip = ImageSequenceClip(output_frames, fps=clip.fps)
        clip.write_videofile(str(output_video_path), audio=False)


if __name__ == "__main__":
    utils.iterate(annotation_path, operation, extension=".xgtf", progress_bar=False)
