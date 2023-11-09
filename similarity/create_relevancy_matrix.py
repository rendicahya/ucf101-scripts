import json
import re
from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm


def encode_cached(phrase, model, embed_bank):
    if phrase not in embed_bank:
        embedding = model.encode(phrase)
        embed_bank[phrase] = embedding

    return embed_bank[phrase]


def calc_similarity(phrase1, phrase2, model, embed_bank):
    emb1 = encode_cached(phrase1, model, embed_bank)
    emb2 = encode_cached(phrase2, model, embed_bank)

    return float(util.cos_sim(emb1, emb2))


dataset_name = "unidet"
dataset_path = Path("/nas.dbms/randy/datasets/ucf101")
classnames_path = Path(f"{dataset_name}-classnames.json")

assert dataset_path.exists() and dataset_path.is_dir()
assert classnames_path.exists() and classnames_path.is_file()

camelcase_tokenizer = re.compile(r"(?<!^)(?=[A-Z])")
n_subdir = sum([1 for f in dataset_path.iterdir() if f.is_dir()])
actions = [action.name for action in dataset_path.iterdir()]
output_dir = Path(f"{dataset_name}-matrix")

output_dir.mkdir(exist_ok=True)

with open(classnames_path, "r") as file:
    classnames = json.load(file)

models = (
    "all-mpnet-base-v2",
    "multi-qa-mpnet-base-dot-v1",
    "all-distilroberta-v1",
    "all-MiniLM-L12-v2",
    "multi-qa-distilbert-cos-v1",
    "all-MiniLM-L6-v2",
    "multi-qa-MiniLM-L6-cos-v1",
    "paraphrase-multilingual-mpnet-base-v2",
    "paraphrase-albert-small-v2",
    "paraphrase-multilingual-MiniLM-L12-v2",
)

for model_name in models:
    print("Running model:", model_name)

    model = SentenceTransformer(model_name)
    embed_bank = {}
    df_data = []

    for subdir in tqdm(dataset_path.iterdir(), total=n_subdir):
        action = camelcase_tokenizer.sub(" ", subdir.name)
        row = [
            calc_similarity(action.lower(), obj.lower(), model, embed_bank)
            for obj in classnames
        ]

        df_data.append(row)

    pd.DataFrame(df_data, columns=classnames, index=actions).to_csv(
        output_dir / f"{model_name}.csv"
    )
