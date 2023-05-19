from pathlib import Path

import numpy as np
from moviepy.editor import VideoFileClip, clips_array, concatenate_videoclips

videos = (
    "/nas.dbms/randy/datasets/ucf101-mix-scene-video-black/Basketball/v_Basketball_g08_c04.mp4",
    "/nas.dbms/randy/datasets/ucf101-mix-scene-video-mean/Basketball/v_Basketball_g08_c04.mp4",
    "/nas.dbms/randy/datasets/ucf101-mix-scene-video-blur/Basketball/v_Basketball_g08_c04.mp4",
    "/nas.dbms/randy/datasets/ucf101-mix-scene-video-inpaint/Basketball/v_Basketball_g08_c04.mp4",
)

output_path = Path("/nas.dbms/randy/datasets/Collage 3.mp4")
clip = [VideoFileClip(str(video)) for video in videos]
clip = np.array(clip).reshape(2, 2)
clip = clips_array(clip)

final_video = concatenate_videoclips([clip])
final_video.without_audio().write_videofile(str(output_path))
