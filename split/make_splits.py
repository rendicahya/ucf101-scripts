import shutil
from pathlib import Path

from tqdm import tqdm


def copy(file_list, dataset_dir, dest_root):
    with open(file_list, "r") as file:
        bar = tqdm(list(file))

        for line in bar:
            file_path = line.split(" ")[0].strip()
            dest_subdir = line.split("/")[0].strip()
            dest_dir = dest_root / dest_subdir

            if not dest_dir.exists():
                dest_dir.mkdir()

            shutil.copy2(dataset_dir / file_path, dest_dir)
            bar.set_description(file_path)


def main():
    dataset_dir = Path("/nas.dbms/randy/datasets/ucf101")

    train_list = Path(
        "/nas.dbms/randy/datasets/ucf101-scripts/split/ucfTrainTestlist/split-train-01.txt"
    )
    val_list = Path(
        "/nas.dbms/randy/datasets/ucf101-scripts/split/ucfTrainTestlist/split-val-01.txt"
    )
    test_list = Path(
        "/nas.dbms/randy/datasets/ucf101-scripts/split/ucfTrainTestlist/testlist01.txt"
    )

    dest_root = Path("/nas.dbms/randy/datasets/ucf101-split")

    train_dir = dest_root / "train"
    val_dir = dest_root / "val"
    test_dir = dest_root / "test"

    shutil.rmtree(train_dir, ignore_errors=True)
    shutil.rmtree(val_dir, ignore_errors=True)
    shutil.rmtree(test_dir, ignore_errors=True)

    train_dir.mkdir(parents=True)
    val_dir.mkdir()
    test_dir.mkdir()

    print("Copying train files...")
    copy(train_list, dataset_dir, train_dir)

    print("Copying validation files...")
    copy(val_list, dataset_dir, val_dir)

    print("Copying test files...")
    copy(test_list, dataset_dir, test_dir)


if __name__ == "__main__":
    main()
