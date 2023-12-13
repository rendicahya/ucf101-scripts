import random
from pathlib import Path

from assertpy.assertpy import assert_that
from python_config import Config

conf = Config("config.json")
dataset_path = Path(conf.generate.file_list.input_dir)
class_index = Path(conf.generate.file_list.class_index)
absolute = conf.generate.file_list.absolute
extension = conf.generate.file_list.ext
indexes = {}

assert_that(dataset_path).is_directory().is_readable()
assert_that(class_index).is_file().is_readable()

with open(class_index, "r") as file:
    for line in file:
        index, name = line.strip().split()
        indexes[name] = int(index)

files = []

for file in dataset_path.rglob("*"):
    if file.is_file() and file.suffix == extension:
        line = str(file) if absolute else str(file.relative_to(dataset_path))
        label = line.split("/")[5 if absolute else 0]
        class_index = str(indexes[label])

        files.append(line + " " + class_index)

if conf.generate.file_list.shuffle:
    random.shuffle(files)

with open(conf.generate.file_list.output, "w") as writer:
    for file in files:
        writer.write(f"{file}\n")
