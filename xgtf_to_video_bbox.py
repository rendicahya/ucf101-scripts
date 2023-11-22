import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Callable

import cv2
import xgtf_parser
from assertpy.assertpy import assert_that
from python_config import Config
from python_file import count_files
from python_video import frames_to_video, video_frames, video_info
from tqdm import tqdm

colors = (
    (0, 0, 255),
    (0, 255, 0),
    (255, 0, 0),
    (0, 255, 255),
    (255, 0, 255),
    (255, 255, 0),
    (255, 255, 255),
)


def draw_bbox(xgtf, video_path, conf):
    people_bbox = xgtf_parser.parse(xgtf)

    if people_bbox is None:
        return

    frames = video_frames(video_path)

    for i, frame in enumerate(frames):
        for person_id, person_bbox in people_bbox.items():
            if conf.video_bbox.output.print_frame_number:
                cv2.putText(
                    frame,
                    str(i + 1),
                    (5, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    colors[0],
                    1,
                    cv2.LINE_AA,
                )

            if i not in person_bbox:
                continue

            x1, y1, w, h = person_bbox[i]
            x2 = x1 + w
            y2 = y1 + h

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                colors[person_id % len(colors)],
                conf.video_bbox.output.bbox_thickness,
            )

        yield frame


def process_video_frame(frames: list, operation: Callable):
    for frame in frames:
        yield operation(frame)


def core_job(xgtf, input_video_path, output_video_path, conf, bar):
    bbox_frames = draw_bbox(xgtf, input_video_path, conf)
    fps = video_info(input_video_path)["fps"]
    grayscale_op = lambda f: cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
    bbox_frames_rgb = process_video_frame(bbox_frames, grayscale_op)

    bar.set_description(xgtf.stem)
    output_video_path.parent.mkdir(parents=True, exist_ok=True)

    frames_to_video(
        bbox_frames_rgb,
        target=output_video_path,
        writer=conf.video_bbox.video_writer,
        fps=fps,
    )

    bar.update(1)


def main():
    conf = Config("config.json")
    xgtf_path = Path(conf.xgtf.path)
    input_path = Path(conf.ucf101.path)
    output_path = Path(conf.video_bbox.output.path)

    assert_that(xgtf_path).is_directory().is_readable()
    assert_that(input_path).is_directory().is_readable()

    n_files = count_files(xgtf_path)
    bar = tqdm(total=n_files)
    n_cores = multiprocessing.cpu_count()

    if conf.video_bbox.multithread:
        print(f"Running jobs on {n_cores} cores...")

    with ThreadPoolExecutor(max_workers=n_cores) as executor:
        futures = []

        for action in xgtf_path.iterdir():
            for xgtf in action.iterdir():
                if xgtf.suffix != ".xgtf":
                    continue

                input_video_path = (
                    input_path / action.name / (xgtf.with_suffix(conf.ucf101.ext).name)
                )

                output_video_path = (
                    output_path / action.name / (xgtf.with_suffix(".mp4").name)
                )

                if conf.video_bbox.multithread:
                    futures.append(
                        executor.submit(
                            partial(
                                core_job,
                                xgtf,
                                input_video_path,
                                output_video_path,
                                conf,
                                bar,
                            )
                        )
                    )
                else:
                    core_job(xgtf, input_video_path, output_video_path, conf, bar)

    bar.close()


if __name__ == "__main__":
    main()
