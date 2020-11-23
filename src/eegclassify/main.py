import logging
from typing import Tuple

import click
import coloredlogs
import pandas as pd
import numpy as np
import sklearn

from . import load, features, preprocess

logger = logging.getLogger(__name__)


@click.group()
def main():
    logging.basicConfig(level=logging.INFO)
    coloredlogs.install(fmt="%(asctime)s %(levelname)s %(name)s %(message)s")


@main.command()
def train():
    """
    Train classifier on data.

        1. Load data
        2. Preprocess
        3. Compute features
        4. Train
    """

    logger.info("Loading data...")
    df = load.load_labeled_eeg2()

    logger.info("Preprocessing...")
    df = _preprocess(df)

    # Filter out categories of interest
    df = df[df["class"].isin(["Editing->Code", "Twitter"])]

    logger.info("Computing features...")
    df = features.compute_features(df)
    X, y = df_to_vectors(df)
    _train(X, y)


def _train(X, y):
    logger.info("Splitting into train and test set")
    # Split into train and test
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, test_size=0.3
    )

    logger.info("Training...")
    clf = sklearn.svm.SVC()
    # clf = sklearn.ensemble.RandomForestClassifier(n_estimators=10)
    clf.fit(X_train, y_train)
    logger.info(f"Test score: {clf.score(X_test, y_test)}")

    y_pred = clf.predict(X_test)
    precision, recall, fbeta, support = sklearn.metrics.precision_recall_fscore_support(
        y_test, y_pred
    )
    logger.info(
        {"precision": precision, "recall": recall, "fbeta": fbeta, "support": support}
    )

    print(sklearn.metrics.confusion_matrix(y_test, y_pred))

    cross_val(clf, X, y, 3)


def _preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses dataframe"""
    min_duration = 5
    df = preprocess.split_rows(df, min_duration)
    df = clean(df)
    return df


def clean(df: pd.DataFrame):
    # TODO: Check signal quality

    sfreq = 250  # NOTE: Will be different for different devices
    bads = []
    for i, row in df.iterrows():
        samples = len(row["raw_data"])
        seconds = samples / sfreq
        duration = row["stop"] - row["start"]
        if seconds < 0.95 * duration.total_seconds():
            logger.warning(
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
    # print(X)

    # Map of categories to codes
    catmap = dict(enumerate(df["class"].cat.categories))
    logger.info(f"Classes: {catmap}")

    # label vector
    y = np.array(df["class"].cat.codes)
    # print(y)

    return X, y


def cross_val(clf, X, y, n):
    logger.info(sklearn.model_selection.cross_val_score(clf, X, y, cv=n))
    logger.info(sklearn.model_selection.cross_val_predict(clf, X, y, cv=n))
    logger.info(sklearn.model_selection.cross_validate(clf, X, y, cv=n))
