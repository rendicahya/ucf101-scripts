import pathlib
import random

import click


@click.command()
@click.argument(
    "input-path",
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
@click.argument("extension", nargs=1, required=True, type=str)
@click.argument(
    "label-file",
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
    "output",
    nargs=1,
    required=True,
    type=click.Path(
        file_okay=True,
        dir_okay=False,
        exists=False,
        writable=True,
        path_type=pathlib.Path,
    ),
    default="list.txt",
)
@click.option("--shuffle", is_flag=True)
@click.option("--absolute", is_flag=True)
def main(input_path, extension, label_file, output, shuffle, absolute):
    indexes = {}

    with open(label_file, "r") as file:
        for line in file:
            line = line.strip().split()
            index, name = line
            indexes[name] = int(index) - 1

    files = []

    for file in input_path.rglob("*"):
        if file.is_file() and file.suffix == extension:
            line = str(file) if absolute else str(file.relative_to(input_path))
            label = line.split("/")[5 if absolute else 0]

            files.append(line + " " + str(indexes[label]))

    if shuffle:
        random.shuffle(files)

    with open(output, "w") as writer:
        for file in files:
            writer.write(f"{file}\n")


if __name__ == "__main__":
    main()
