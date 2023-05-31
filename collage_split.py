from pathlib import Path

import numpy as np
from moviepy.editor import VideoFileClip, clips_array, concatenate_videoclips

input_path = Path("/nas.dbms/randy/datasets/ucf101-segmentation")
output_path = Path("/nas.dbms/randy/datasets/ucf101-segmentation-collage")


for split in input_path.iterdir():
    (output_path / split.name).mkdir(parents=True, exist_ok=True)

    for action in split.iterdir():
        count = 0
        clips = []
        clip = []

        for video in action.iterdir():
            count += 1

            clip.append(VideoFileClip(str(video)))

            if count % 6 == 0:
                clip = np.array(clip).reshape(2, 3)
                clip = clips_array(clip)

                clips.append(clip)

                clip = []

        collage = concatenate_videoclips(clips)
        collage.without_audio().write_videofile(
            str(output_path / split.name / action.with_suffix(".mp4").name)
        )
