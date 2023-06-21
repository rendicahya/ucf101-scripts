import pathlib

import click


@click.command()
@click.argument(
    "dataset-path",
    nargs=1,
    required=True,
    type=click.Path(
        file_okay=False,
        dir_okay=True,
        exists=True,
        readable=True,
        path_type=pathlib.Path,
    ),
)
@click.argument(
    "train-list",
    nargs=1,
    required=True,
    type=click.Path(
        file_okay=True,
        dir_okay=False,
        exists=True,
        readable=True,
        path_type=pathlib.Path,
    ),
)
@click.argument(
    "test-list",
    nargs=1,
    required=True,
    type=click.Path(
        file_okay=True,
        dir_okay=False,
        exists=True,
        readable=True,
        path_type=pathlib.Path,
    ),
)
@click.argument(
    "annotation-path",
    nargs=1,
    required=True,
    type=click.Path(
        file_okay=False,
        dir_okay=True,
        exists=True,
        readable=True,
        path_type=pathlib.Path,
    ),
)
def main(dataset_path, train_list, test_list, annotation_path):
    train = open(train_list).readlines()
    train = [
        line.strip().replace(".", " ").replace("/", " ").split()[1] for line in train
    ]
    train_size = len(train)

    test = open(test_list).readlines()
    test = [
        line.strip().replace(".", " ").replace("/", " ").split()[1] for line in test
    ]
    test_size = len(test)

    anno_train_count = 0
    anno_test_count = 0

    for action in annotation_path.iterdir():
        for anno in action.iterdir():
            if anno.suffix != ".xgtf":
                continue

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


if __name__ == "__main__":
    main()
