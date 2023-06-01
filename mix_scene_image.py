import random
from pathlib import Path

import cv2
import utils
from bs4 import BeautifulSoup
from moviepy.editor import ImageSequenceClip, VideoFileClip

annotation_path = Path("/nas.dbms/randy/projects/ucf101-scripts/annotations")
video_path = Path("/nas.dbms/randy/datasets/ucf101")
scene_root_path = Path("/nas.dbms/randy/datasets/places365")
output_path = Path("/nas.dbms/randy/datasets/ucf101-mix-scene-image")

with open(scene_root_path / "val.txt") as file:
    scene_list = [line.strip() for line in file]

action_only = False

colors = (
    (0, 0, 255),
    (0, 255, 0),
    (255, 0, 0),
    (0, 255, 255),
    (255, 0, 255),
    (255, 255, 0),
    (255, 255, 255),
)


def parse_annotation(file):
    with open(file) as f:
        try:
            soup = BeautifulSoup(f, features="xml")
        except:
            return None

    data = soup.find("data")
    people_bbox = {}

    if data is None:
        return None

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

    return people_bbox


def operation(action, anno_file):
    people_bbox = parse_annotation(anno_file)

    if not people_bbox:
        return

    input_video_path = video_path / action.name / (anno_file.with_suffix(".avi").name)
    output_video_path = output_path / action.name / (anno_file.with_suffix(".mp4").name)

    output_video_path.parent.mkdir(parents=True, exist_ok=True)

    clip = VideoFileClip(str(input_video_path))
    output_frames = []

    scene_pick = random.choice(scene_list)
    scene_path = Path(scene_root_path / scene_pick)
    scene_image = cv2.imread(str(scene_path))
    scene_image = cv2.resize(scene_image, (clip.w, clip.h))

    for i, frame in enumerate(clip.iter_frames()):
        mix_image = scene_image.copy()

        for person_id, person_bbox in people_bbox.items():
            if i not in person_bbox:
                continue

            x1, y1, w, h = person_bbox[i]
            x2 = x1 + w
            y2 = y1 + h
            person_crop = frame[y1:y2, x1:x2]
            mix_image[y1:y2, x1:x2] = person_crop

        output_frames.append(mix_image)

    if len(output_frames) > 0:
        clip = ImageSequenceClip(output_frames, fps=clip.fps)
        clip.without_audio().write_videofile(str(output_video_path))


if __name__ == "__main__":
    utils.iterate(annotation_path, operation, extension=".xgtf", single=True)
