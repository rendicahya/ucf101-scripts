from pathlib import Path

from moviepy.editor import VideoFileClip
from PIL import Image
from tqdm import tqdm
from utils import count_files

if __name__ == "__main__":
    input_path = Path("/nas.dbms/randy/datasets/ucf101")
    output_path = Path("/nas.dbms/randy/datasets/ucf101-frames")

    assert input_path.exists() and input_path.is_dir()

    ext = "avi"
    n_files = count_files(input_path, recursive=True, extension=f".{ext}")

    with tqdm(total=n_files) as bar:
        for action in input_path.iterdir():
            for file in action.iterdir():
                clip = VideoFileClip(str(file))
                frames_output_path = output_path / action.name / file.name

                bar.set_description(file.name)
                frames_output_path.mkdir(parents=True, exist_ok=True)

                for i, frame in enumerate(clip.iter_frames()):
                    frame_full_path = frames_output_path / f"img_{i+1:05d}.jpg"
                    image = Image.fromarray(frame)

                    image.save(frame_full_path)

                clip.close()
                bar.update(1)
