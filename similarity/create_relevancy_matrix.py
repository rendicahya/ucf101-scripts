import json
import pathlib
import re

import click
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
def main(dataset_path):
    camelcase_tokenizer = re.compile(r"(?<!^)(?=[A-Z])")
    n_subdir = sum([1 for _ in dataset_path.iterdir() if _.is_dir()])
    actions = [action.name for action in dataset_path.iterdir()]

    with open("objects365.json", "r") as file:
        obj365 = json.load(file)

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
            action = camelcase_tokenizer.sub(" ", subdir.name).lower()
            row = [calc_similarity(action, obj, model, embed_bank) for obj in obj365]

            df_data.append(row)

        pd.DataFrame(df_data, columns=obj365, index=actions).to_csv(
            f"matrix/{model_name}.csv"
        )


if __name__ == "__main__":
    main()
