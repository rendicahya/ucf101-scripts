import random
from pathlib import Path

import numpy as np
import skimage
import utils
from bs4 import BeautifulSoup
from moviepy.editor import ImageSequenceClip, VideoFileClip
from skimage.filters import gaussian
from skimage.restoration import inpaint

random.seed(46)

available_modes = "mean", "blur", "black", "inpaint"
scene_bbox_mode = available_modes[2]

annotation_path = Path("/nas.dbms/randy/projects/ucf101-scripts/annotations")
video_path = Path("/nas.dbms/randy/datasets/ucf101")
output_path = Path(
    f"/nas.dbms/randy/datasets/ucf101-mix-scene-video-XYZ-{scene_bbox_mode}"
)

crop_action_temporally = False
scene_mean_temporal_smoothing = 5
scene_blur_sigma = 25

with open("annotation-list.txt") as file:
    anno_list = [line.strip() for line in file]

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

                if crop_action_temporally:
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


def operation(action, actor_anno):
    actor_bbox = parse_annotation(actor_anno)

    if not actor_bbox:
        return

    actor_video_path = video_path / action.name / actor_anno.with_suffix(".avi").name

    actor_video = VideoFileClip(str(actor_video_path))
    actor_frames = actor_video.iter_frames()

    scene_anno = random.choice(anno_list)
    scene_anno_path = Path(scene_anno)
    scene_video_path = (
        video_path
        / scene_anno_path.parent.name
        / scene_anno_path.with_suffix(".avi").name
    )
    scene_video = VideoFileClip(str(scene_video_path))
    scene_frames = list(scene_video.iter_frames())
    scene_bbox = parse_annotation(annotation_path / scene_anno)
    n_scene_frames = len(scene_frames)

    scene_bbox_temporal_cache = []
    output_frames = []

    for i, actor_frame in enumerate(actor_frames):
        canvas = scene_frames[i % n_scene_frames].copy()

        if canvas.shape[0] != actor_video.h or canvas.shape[1] != actor_video.w:
            canvas = skimage.transform.resize(
                canvas,
                (actor_video.h, actor_video.w),
                mode="reflect",
                preserve_range=True,
                anti_aliasing=True,
            )

        mask = np.zeros(canvas.shape[:-1], dtype=bool)

        for person_id, person_bbox in scene_bbox.items():
            i_mod = i % n_scene_frames

            if i_mod not in person_bbox:
                continue

            x1, y1, w, h = person_bbox[i_mod]
            x2 = x1 + w
            y2 = y1 + h

            if scene_bbox_mode == "mean":
                bbox_crop = canvas[y1:y2, x1:x2]
                bbox_color = np.mean(bbox_crop, axis=(0, 1))
                scene_bbox_temporal_cache.append(bbox_color)

                if len(scene_bbox_temporal_cache) > scene_mean_temporal_smoothing:
                    scene_bbox_temporal_cache.pop(0)

                bbox_replace = np.mean(scene_bbox_temporal_cache, axis=0)
                canvas[y1:y2, x1:x2] = bbox_replace
            elif scene_bbox_mode == "blur":
                bbox_crop = canvas[y1:y2, x1:x2]
                bbox_blur = gaussian(bbox_crop, sigma=scene_blur_sigma, channel_axis=-1)
                bbox_blur *= 255
                bbox_blur = bbox_blur.astype(np.uint8)
                canvas[y1:y2, x1:x2] = bbox_blur
            elif scene_bbox_mode == "black":
                canvas[y1:y2, x1:x2] = 0
            elif scene_bbox_mode == "inpaint":
                canvas[y1:y2, x1:x2] = 0
                mask[y1:y2, x1:x2] = 1

        if scene_bbox_mode == "inpaint":
            canvas = inpaint.inpaint_biharmonic(canvas, mask, channel_axis=-1)
            canvas *= 255
            canvas = canvas.astype(np.uint8)

        for person_id, person_bbox in actor_bbox.items():
            if i not in person_bbox:
                continue

            x1, y1, w, h = person_bbox[i]
            x2 = x1 + w
            y2 = y1 + h
            canvas[y1:y2, x1:x2] = actor_frame[y1:y2, x1:x2]

        output_frames.append(canvas)

    output_video_path = output_path / action.name / actor_anno.with_suffix(".mp4").name

    output_video_path.parent.mkdir(parents=True, exist_ok=True)

    if len(output_frames) > 0:
        clip = ImageSequenceClip(output_frames, fps=actor_video.fps)
        clip.write_videofile(str(output_video_path), audio=False)


if __name__ == "__main__":
    utils.iterate(
        annotation_path, operation, extension=".xgtf", progress_bar=False, single=True
    )
