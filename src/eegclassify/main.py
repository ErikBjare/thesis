import logging
from typing import Tuple

import click
import pandas as pd
import numpy as np
import sklearn
from sklearn import svm

from . import load
from . import features

logger = logging.getLogger(__name__)


@click.group()
def main():
    logging.basicConfig(level=logging.INFO)


@main.command()
def train():
    df = load.load_labeled_eeg2()
    print(df)
    df = features.compute_features(df)
    print(df.describe(datetime_is_numeric=True))

    df = clean(df)
    print(df)

    # Filter out categories of interest
    df = df[(df["class"] == "Programming") | (df["class"] == "Twitter")]

    X, y = df_to_vectors(df)
    cross_val_svm(X, y, 2)


def clean(df: pd.DataFrame):
    # TODO: Check signal quality

    sfreq = 250  # NOTE: Will be different for different devices
    bads = []
    for i, row in df.iterrows():
        samples = len(row["raw_data"])
        seconds = samples / sfreq
        duration = row["stop"] - row["start"]
        if seconds < 0.95 * duration.total_seconds():
            print(
                f"Bad row found, only had {seconds}s of data out of {duration.total_seconds()}s"
            )
            bads.append(i)

    return df.drop(bads)


@main.command()
def clear():
    """Clears the cache"""
    load.memory.clear()


def df_to_vectors(df: pd.DataFrame) -> Tuple[np.array, np.array]:
    # feature vector
    X = np.array(
        [
            [band for channel in row for band in channel[1:]]
            for row in df["bandpower"].values.tolist()
        ]
    )
    print(X)

    # Map of categories to codes
    catmap = dict(enumerate(df["class"].cat.categories))
    print(catmap)

    # label vector
    y = np.array(df["class"].cat.codes)
    print(y)

    return X, y


def cross_val_svm(X, y, n):
    clf = svm.SVC()
    print(sklearn.model_selection.cross_val_score(clf, X, y, cv=n))
    print(sklearn.model_selection.cross_val_predict(clf, X, y, cv=n))
    print(sklearn.model_selection.cross_validate(clf, X, y, cv=n))
