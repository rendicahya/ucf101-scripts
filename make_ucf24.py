from pathlib import Path
from shutil import copy2

from python_config import Config
from python_file import count_files
from tqdm import tqdm

conf = Config("config.json")
ucf101_dir = Path(conf.ucf101.path)
ucf101_ext = conf.ucf101.ext
ucf24_dir = Path(conf.ucf24.path)
xgtf_dir = Path(conf.xgtf.path)
n_files = count_files(xgtf_dir)
bar = tqdm(total=n_files)
count = 0

for action in xgtf_dir.iterdir():
    for xgtf in action.iterdir():
        source_path = ucf101_dir / action.name / xgtf.with_suffix(ucf101_ext).name
        target_path = ucf24_dir / action.name

        if not source_path.exists():
            continue

        target_path.mkdir(exist_ok=True, parents=True)
        bar.set_description(xgtf.stem)
        copy2(source_path, target_path)
        bar.update(1)

        count += 1

bar.close()
print(f"Copied {count} of {n_files} files")
