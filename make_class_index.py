from pathlib import Path

from assertpy.assertpy import assert_that
from python_config import Config

conf = Config("config.json")
dataset_path = Path(conf.generate.class_index.input_dir)
output = conf.generate.class_index.output
index = 1

assert_that(dataset_path).is_directory().is_readable()

with open(output, "w") as writer:
    for action in dataset_path.iterdir():
        writer.write(f"{index} {action.name}\n")

        index += 1
