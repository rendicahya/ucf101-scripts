from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    input = Path("ucfTrainTestlist/trainlist01.txt")
    train_output = Path("ucfTrainTestlist/split-train-01.txt")
    val_output = Path("ucfTrainTestlist/split-val-01.txt")
    ratio = 0.8
    random_state = 42

    df = pd.read_csv(input, header=None)
    train, val = train_test_split(
        df, train_size=ratio, random_state=random_state, shuffle=False
    )

    train.to_csv(train_output, header=False, index=False)
    val.to_csv(val_output, header=False, index=False)


if __name__ == "__main__":
    main()
