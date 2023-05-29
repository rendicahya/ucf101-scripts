from contextlib import nullcontext
from pathlib import Path

from tqdm import tqdm


def iterate(path: Path, operation, extension=None, progress_bar=True, single=False):
    n_files = count_files(path, recursive=True, extension=extension)

    with tqdm(total=n_files) if progress_bar else nullcontext() as bar:
        for action in path.iterdir():
            for video in action.iterdir():
                if progress_bar:
                    bar.set_description(video.name[:30])
                    bar.update(1)

                operation(action, video)

                if single:
                    break

            if single:
                break


def count_files(path: Path, recursive=False, extension=None):
    pattern = "**/*" if recursive else "*"
    filter = (
        (lambda f: f.is_file())
        if extension is None
        else (lambda f: f.is_file() and f.suffix == extension)
    )

    return sum(1 for f in path.glob(pattern) if filter(f))
