from pathlib import Path

import numpy as np
from python_config import Config
from assertpy.assertpy import assert_that
import xgtf_parser
from python_video import video_info
from python_file import count_files
import cv2
from tqdm import tqdm


def main():
    conf = Config("config.json")
    xgtf_dir = Path(conf.xgtf.path)
    ucf101_dir = Path(conf.ucf101.path)
    output_dir = Path(conf.mask.path)
    bar = tqdm(total=count_files(xgtf_dir))

    assert_that(xgtf_dir).is_directory().is_readable()
    assert_that(ucf101_dir).is_directory().is_readable()

    for action in xgtf_dir.iterdir():
        for xgtf in action.iterdir():
            bar.set_description(xgtf.name)

            if xgtf.suffix != ".xgtf":
                continue

            people_bbox = xgtf_parser.parse(xgtf, action_only=conf.mask.action_only)

            if not people_bbox:
                continue

            video_path = ucf101_dir / action.name / (xgtf.with_suffix(".avi").name)
            output_path = output_dir / action.name / xgtf.stem

            output_path.mkdir(parents=True, exist_ok=True)

            info = video_info(video_path)
            n_frames = info["n_frames"]
            width, height = info["width"], info["height"]

            for i in range(n_frames):
                mask = np.zeros((height, width), dtype=np.uint8)

                for person_id, person_bbox in people_bbox.items():
                    if i not in person_bbox:
                        continue

                    x1, y1, w, h = person_bbox[i]
                    x2 = x1 + w
                    y2 = y1 + h
                    mask[y1:y2, x1:x2] = 255

                cv2.imwrite(str(output_path / f"{i:04}.png"), mask)

            bar.update(1)


if __name__ == "__main__":
    main()
