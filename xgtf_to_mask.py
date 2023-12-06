from pathlib import Path

import cv2
import numpy as np
import xgtf_parser
from assertpy.assertpy import assert_that
from python_config import Config
from python_file import count_files
from python_video import video_info
from tqdm import tqdm


def main():
    conf = Config("config.json")
    xgtf_dir = Path(conf.xgtf.path)
    ucf101_dir = Path(conf.ucf101.path)
    ucf101_ext = conf.ucf101.ext
    mask_ext = conf.mask.ext
    output_root = Path(conf.mask.path)
    bar = tqdm(total=count_files(xgtf_dir))
    n_digits = conf.mask.n_digits

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

            video_path = ucf101_dir / action.name / (xgtf.with_suffix(ucf101_ext).name)
            output_dir = output_root / action.name / xgtf.stem

            output_dir.mkdir(parents=True, exist_ok=True)

            info = video_info(video_path)
            n_frames = info["n_frames"]
            width, height = info["width"], info["height"]

            for i in range(n_frames):
                mask = np.zeros((height, width), dtype=np.uint8)
                output_file = output_dir / (f"%0{n_digits}d{mask_ext}" % i)

                for person_id, person_bbox in people_bbox.items():
                    if i not in person_bbox:
                        continue

                    x1, y1, w, h = person_bbox[i]
                    x2 = x1 + w
                    y2 = y1 + h
                    mask[y1:y2, x1:x2] = 255

                cv2.imwrite(str(output_file), mask)

            bar.update(1)


if __name__ == "__main__":
    main()
