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


def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [
        token for token in tokens if token.isalnum() and token not in stop_words
    ]

    return filtered_tokens


def sim_nltk(phrase1, phrase2):
    tokens1 = preprocess_text(phrase1)
    tokens2 = preprocess_text(phrase2)

    synsets1 = [lesk(tokens1, token) for token in tokens1]
    synsets2 = [lesk(tokens2, token) for token in tokens2]

    similarity_score = 0
    count = 0

    for synset1 in synsets1:
        if synset1 is None:
            continue

        for synset2 in synsets2:
            if synset2 is None:
                continue

            similarity = synset1.path_similarity(synset2)

            if similarity is not None:
                similarity_score += similarity
                count += 1

    if count > 0:
        similarity_score /= count

    return similarity_score


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
            f"matrices/relevancy-matrix-{model}.csv"
        )


if __name__ == "__main__":
    main()
