import json
import pathlib
import re

import click
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.wsd import lesk
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm


def calc_similarity(phrase1, phrase2, model):
    emb1 = model.encode(phrase1)
    emb2 = model.encode(phrase2)

    return util.cos_sim(emb1, emb2)


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

    for model in models:
        print("Running model:", model)

        sbert_model = SentenceTransformer(model)
        df_data = []

        for subdir in tqdm(dataset_path.iterdir(), total=n_subdir):
            action = camelcase_tokenizer.sub(" ", subdir.name).lower()
            row = [float(calc_similarity(action, obj, sbert_model)) for obj in obj365]

            df_data.append(row)

        pd.DataFrame(df_data, columns=obj365, index=actions).to_csv(
            f"matrix/{model}.csv"
        )


if __name__ == "__main__":
    main()
