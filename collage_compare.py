from pathlib import Path

import numpy as np
from moviepy.editor import VideoFileClip, clips_array, concatenate_videoclips

orig_video_path = Path("/nas.dbms/randy/datasets/ucf101")
proc_video_path = Path("/nas.dbms/randy/datasets/ucf101-scenes")
output_path = Path("/nas.dbms/randy/datasets/ucf101-scenes-compare")

output_path.mkdir(parents=True, exist_ok=True)

for action in proc_video_path.iterdir():
    count = 0
    clips = []
    pairs = []

    for video in action.iterdir():
        orig_video = orig_video_path / action.name / video.with_suffix(".avi").name
        count += 1

        pairs.append([VideoFileClip(str(orig_video)), VideoFileClip(str(video))])

        if count % 3 == 0:
            clip = np.array(pairs).T
            clip = clips_array(clip)
            pairs = []

            clips.append(clip)

    final_video = concatenate_videoclips(clips)
    final_video.without_audio().write_videofile(
        str(output_path / action.with_suffix(".mp4").name)
    )
