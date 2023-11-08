import random
from pathlib import Path

if __name__ == "__main__":
    dataset_path = Path("/nas.dbms/randy/datasets/ucf101")
    label_file = Path("split/ucfTrainTestlist/classInd.txt")
    shuffle = False
    absolute = True
    extension = ".avi"
    output = "ucf101-list.txt"
    indexes = {}

    assert dataset_path.exists()
    assert dataset_path.is_dir()
    assert label_file.exists()
    assert label_file.is_file()

    with open(label_file, "r") as file:
        for line in file:
            line = line.strip().split()
            index, name = line
            indexes[name] = int(index) - 1

    files = []

    for file in dataset_path.rglob("*"):
        if file.is_file() and file.suffix == extension:
            line = str(file) if absolute else str(file.relative_to(dataset_path))
            label = line.split("/")[5 if absolute else 0]
            class_index = str(indexes[label])

            files.append(line + " " + class_index)

    if shuffle:
        random.shuffle(files)

    with open(output, "w") as writer:
        for file in files:
            writer.write(f"{file}\n")
