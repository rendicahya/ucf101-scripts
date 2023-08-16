"""
This script reads YOLOv5 annotations and does intra-video augmentations.
"""

import pathlib
import random

import albumentations as A
import click
from albumentations.augmentations.blur.transforms import *
from albumentations.augmentations.geometric.rotate import *
from albumentations.augmentations.geometric.transforms import *
from albumentations.augmentations.transforms import *
from albumentations.core.transforms_interface import *
from moviepy.editor import ImageSequenceClip, VideoFileClip


def yolo_to_pixel(yolo_annotation, image_width, image_height):
    x_center, y_center, width, height = yolo_annotation

    x = int((x_center - width / 2) * image_width)
    y = int((y_center - height / 2) * image_height)
    w = int(width * image_width)
    h = int(height * image_height)

    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h

    return x1, y1, x2, y2


def augment(image):
    avail_methods = {
        "InvertImg": InvertImg,
        "Flip": Flip,
        "VerticalFlip": VerticalFlip,
        "HorizontalFlip": HorizontalFlip,
        "CLAHE": CLAHE,
        "GaussianBlur": GaussianBlur,
        "Perspective": Perspective,
        "Sharpen": A.augmentations.transforms.Sharpen,
        "RandomBrightnessContrast": RandomBrightnessContrast,
        "Equalize": Equalize,
        "NoOp": NoOp,
        "Transpose": Transpose,
        "Perspective": Perspective,
        "ElasticTransform": ElasticTransform,
        "GridDistortion": GridDistortion,
        "ShiftScaleRotate": ShiftScaleRotate,
    }

    key = random.choice(list(avail_methods.values()))
    key = 1
    transform = A.Compose(
        [
            avail_methods["HorizontalFlip"](p=1),
        ]
    )
    transformed = transform(image=image)

    return transformed["image"]


@click.command()
@click.argument(
    "dataset-path",
    nargs=1,
    required=True,
    type=click.Path(
        file_okay=False,
        dir_okay=True,
        exists=True,
        readable=True,
        path_type=pathlib.Path,
    ),
)
@click.argument(
    "scene-path",
    nargs=1,
    required=True,
    type=click.Path(
        file_okay=False,
        dir_okay=True,
        exists=True,
        readable=True,
        path_type=pathlib.Path,
    ),
)
@click.argument(
    "yolo-anno-path",
    nargs=1,
    required=True,
    type=click.Path(
        file_okay=False,
        dir_okay=True,
        exists=True,
        readable=True,
        path_type=pathlib.Path,
    ),
)
@click.argument(
    "output-path",
    nargs=1,
    required=True,
    type=click.Path(
        file_okay=False,
        dir_okay=True,
        path_type=pathlib.Path,
    ),
)
def main(dataset_path, scene_path, yolo_anno_path, output_path):
    for action in dataset_path.iterdir():
        for file in action.iterdir():
            video = VideoFileClip(str(file))
            video_frames = video.iter_frames()

            scene_video_path = scene_path / action.name / file.name
            scene_video = VideoFileClip(str(scene_video_path))
            scene_frames = scene_video.iter_frames()

            anno_dir = yolo_anno_path / action.name / file.stem
            annotations = {}
            output_frames = []
            output_video_path = output_path / action.name / file.name

            output_video_path.parent.mkdir(parents=True, exist_ok=True)

            for anno_file in anno_dir.iterdir():
                with open(anno_file, "r") as f:
                    values = [
                        list(map(float, line.strip().split()[1:-1])) for line in f
                    ]

                    annotations.update({int(anno_file.stem): values})

            for i, (frame, scene) in enumerate(
                zip(video_frames, reversed(scene_frames))
            ):
                scene_aug = augment(scene)

                if i not in annotations.keys():
                    continue

                for box in annotations[i]:
                    x1, y1, x2, y2 = yolo_to_pixel(box, video.w, video.h)
                    scene_aug[y1:y2, x1:x2] = frame[y1:y2, x1:x2]

                output_frames.append(scene_aug)

            ImageSequenceClip(output_frames, fps=video.fps).write_videofile(
                str(output_video_path), audio=False
            )


if __name__ == "__main__":
    main()
