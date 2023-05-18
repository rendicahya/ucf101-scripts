from pathlib import Path

extensions = (".xgtf",)
path = Path("/nas.dbms/randy/projects/ucf101-scripts/annotations")
files = [
    str(f.relative_to(path))
    for f in path.rglob("*")
    if f.is_file() and f.suffix in extensions
]

with open("list.txt", "w") as writer:
    for file in files:
        writer.write(f"{file}\n")
