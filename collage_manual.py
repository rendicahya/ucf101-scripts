import numpy as np
from moviepy.editor import VideoFileClip, clips_array, concatenate_videoclips

videos = (
    "/nas.dbms/randy/datasets/ucf101-mix-scene-video-black/Basketball/v_Basketball_g08_c04.mp4",
    "/nas.dbms/randy/datasets/ucf101-mix-scene-video-mean/Basketball/v_Basketball_g08_c04.mp4",
    "/nas.dbms/randy/datasets/ucf101-mix-scene-video-blur/Basketball/v_Basketball_g08_c04.mp4",
    "/nas.dbms/randy/datasets/ucf101-mix-scene-video-inpaint/Basketball/v_Basketball_g08_c04.mp4",
)

collage_shape = 2, 2
output_path = "/nas.dbms/randy/datasets/Collage 3.mp4"

clips = [VideoFileClip(v) for v in videos]
collage = np.array(clips).reshape(collage_shape)
collage = clips_array(clips)

final_video = concatenate_videoclips([collage])
final_video.without_audio().write_videofile(output_path)
