from pathlib import Path

from assertpy.assertpy import assert_that
from python_config import Config

conf = Config("config.json")
dataset_path = Path(conf.ucf101.path)
train_list = Path("split/ucfTrainTestlist/trainlist01.txt")
test_list = Path("split/ucfTrainTestlist/testlist01.txt")
xgtf_path = Path(conf.xgtf.path)

assert_that(dataset_path).is_directory().is_readable()
assert_that(xgtf_path).is_directory().is_readable()
assert_that(train_list).is_file().is_readable()
assert_that(test_list).is_file().is_readable()

train = [
    line.strip().replace(".", " ").replace("/", " ").split()[1]
    for line in open(train_list).readlines()
]

test = [
    line.strip().replace(".", " ").replace("/", " ").split()[1]
    for line in open(test_list).readlines()
]

train_size, test_size = len(train), len(test)
xgtf_n_actions = sum(1 for d in xgtf_path.iterdir())

anno_train_count = 0
anno_test_count = 0

for anno in xgtf_path.glob("**/*.xgtf"):
    if anno.stem in train:
        anno_train_count += 1
    elif anno.stem in test:
        anno_test_count += 1

print("Dataset size:", train_size + test_size)
print("Train size:", train_size)
print("Test size:", test_size)
print("Annotations:", anno_train_count + anno_test_count)
print("Annotations in train split:", anno_train_count)
print("Annotations in test split:", anno_test_count)
print("No. of annotated actions class:", xgtf_n_actions)
