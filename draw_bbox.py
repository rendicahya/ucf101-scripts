from pathlib import Path

import cv2
from bs4 import BeautifulSoup
from moviepy.editor import ImageSequenceClip

annotation_path = Path("/nas.dbms/randy/projects/ucf101-scripts/annotations")
dataset_path = Path("/nas.dbms/randy/datasets/ucf101")
output_path = Path("/nas.dbms/randy/datasets/ucf101-bbox-3")
bbox_thickness = 2
n_xml = sum(
    1 for f in annotation_path.glob("**/*") if f.is_file() and f.name.endswith(".xgtf")
)

print_frame_number = False
action_only = True

colors = (
    (0, 0, 255),
    (0, 255, 0),
    (255, 0, 0),
    (0, 255, 255),
    (255, 0, 255),
    (255, 255, 0),
    (255, 255, 255),
)

for action in annotation_path.iterdir():
    for anno_file in action.iterdir():
        if not anno_file.suffix == ".xgtf":
            continue

        print(anno_file.name)

        with open(anno_file) as f:
            try:
                soup = BeautifulSoup(f, features="xml")
            except:
                continue

        data = soup.find("data")
        people_bbox = {}

        if data is None:
            continue

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

                act_start, act_end = [
                    int(i) for i in person_action["framespan"].split(":")
                ]

                for bbox in person_locations.find_all("data:bbox"):
                    start, end = [int(i) for i in bbox["framespan"].split(":")]

                    if action_only:
                        if (
                            act_start <= start <= act_end
                            or act_start <= end <= act_end
                        ):
                            start = max(start, act_start)
                            end = min(end, act_end)

                            for frame in range(start - 1, end):
                                bbox_data = {
                                    frame: (
                                        bbox["x"],
                                        bbox["y"],
                                        bbox["width"],
                                        bbox["height"],
                                    )
                                }

                                person_bbox.update(bbox_data)
                    else:
                        for frame in range(start - 1, end):
                            bbox_data = {
                                frame: (
                                    bbox["x"],
                                    bbox["y"],
                                    bbox["width"],
                                    bbox["height"],
                                )
                            }

                            person_bbox.update(bbox_data)

                people_bbox.update({person_id: person_bbox})

        input_video_path = (
            dataset_path / action.name / (anno_file.with_suffix(".avi").name)
        )
        output_video_path = (
            output_path / action.name / (anno_file.with_suffix(".mp4").name)
        )

        output_video_path.parent.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(input_video_path))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        output_frames = []
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            for person_id, person_bbox in people_bbox.items():
                if print_frame_number:
                    cv2.putText(
                        frame,
                        str(frame_idx + 1),
                        (5, 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        colors[0],
                        1,
                        cv2.LINE_AA,
                    )

                if frame_idx not in person_bbox:
                    continue

                bbox = person_bbox[frame_idx]
                x1, y1, w, h = [int(i) for i in bbox]
                x2 = x1 + w
                y2 = y1 + h

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    colors[person_id % len(colors)],
                    bbox_thickness,
                )

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_idx += 1

            output_frames.append(frame)

        if len(output_frames) > 0:
            clip = ImageSequenceClip(output_frames, fps=fps)
            clip.write_videofile(str(output_video_path), audio=False)
