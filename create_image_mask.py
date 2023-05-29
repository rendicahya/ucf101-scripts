from pathlib import Path

import numpy as np
import utils
from bs4 import BeautifulSoup
from moviepy.editor import VideoFileClip
from PIL import Image

annotation_path = Path("/nas.dbms/randy/projects/ucf101-scripts/annotations")
input_path = Path("/nas.dbms/randy/datasets/ucf101")
output_path = Path("/nas.dbms/randy/datasets/ucf101-mask")
action_only = False
mask_actor = True

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
                                    int(bbox["x"]),
                                    int(bbox["y"]),
                                    int(bbox["width"]),
                                    int(bbox["height"]),
                                )
                            }

                            person_bbox.update(bbox_data)
                else:
                    for frame in range(start - 1, end):
                        bbox_data = {
                            frame: (
                                int(bbox["x"]),
                                int(bbox["y"]),
                                int(bbox["width"]),
                                int(bbox["height"]),
                            )
                        }

                        person_bbox.update(bbox_data)

            people_bbox.update({person_id: person_bbox})

    input_video_path = input_path / action.name / (anno_file.with_suffix(".avi").name)
    output_video_path = output_path / action.name / (anno_file.with_suffix(".avi").name)

    output_video_path.mkdir(parents=True, exist_ok=True)

    clip = VideoFileClip(str(input_video_path))

    for i in range(clip.reader.nframes):
        mask = np.full((clip.h, clip.w), 0 if mask_actor else 255, dtype=np.uint8)

        for person_id, person_bbox in people_bbox.items():
            if i not in person_bbox:
                continue

            x1, y1, w, h = person_bbox[i]
            x2 = x1 + w
            y2 = y1 + h
            mask[y1:y2, x1:x2] = 255 if mask_actor else 0

        Image.fromarray(mask).save(str(output_video_path / f"{i:04}.png"))


if __name__ == "__main__":
    utils.iterate(annotation_path, operation, extension=".xgtf")
