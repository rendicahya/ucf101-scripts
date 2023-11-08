from pathlib import Path

if __name__ == "__main__":
    dataset_path = Path("/nas.dbms/randy/datasets/ucf101")
    train_list = Path("split/ucfTrainTestlist/trainlist01.txt")
    test_list = Path("split/ucfTrainTestlist/testlist01.txt")
    annotation_path = Path("annotations")

    assert dataset_path.exists() and dataset_path.is_dir()
    assert train_list.exists() and train_list.is_file()
    assert test_list.exists() and test_list.is_file()
    assert annotation_path.exists() and annotation_path.is_dir()

    train = [
        line.strip().replace(".", " ").replace("/", " ").split()[1]
        for line in open(train_list).readlines()
    ]

    test = [
        line.strip().replace(".", " ").replace("/", " ").split()[1]
        for line in open(test_list).readlines()
    ]

    train_size, test_size = len(train), len(test)

    anno_train_count = 0
    anno_test_count = 0

    for anno in annotation_path.glob("**/*.xgtf"):
        if anno.stem in train:
            anno_train_count += 1
        elif anno.stem in test:
            anno_test_count += 1

    print("Dataset size:", train_size + test_size)
    print("Train size:", train_size)
    print("Test size:", test_size)
    print("Annotations:", anno_train_count + anno_test_count)
    print("Annotations in train:", anno_train_count)
    print("Annotations in test:", anno_test_count)
