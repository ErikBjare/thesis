import logging
from pathlib import Path
from typing import Tuple, Optional, Dict
from datetime import datetime, timezone
from pprint import pprint

import click
import coloredlogs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

import sklearn
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

import xgboost as xgb

from pyriemann.estimation import Covariances, ERPCovariances, XdawnCovariances
from pyriemann.spatialfilters import CSP
from pyriemann.tangentspace import TangentSpace
from pyriemann.classification import MDM

from . import load, features, preprocess
from .clean import clean

logger = logging.getLogger(__name__)


@click.group()
def main():
    logging.basicConfig(level=logging.INFO)
    coloredlogs.install(fmt="%(asctime)s %(levelname)s %(name)s %(message)s")


@main.command()
@click.option("--use-cache", is_flag=True)
@click.option("--raw", is_flag=True)
@click.option("--since", type=click.DateTime(["%Y-%m-%d"]))
def train(use_cache: bool, raw: bool, since: datetime):
    """
    Train classifier on data.

        1. Load data
        2. Preprocess
        3. Compute features
        4. Train
    """
    df = _load(use_cache, since=since.replace(tzinfo=timezone.utc) if since else None)

    # classdistribution(df)

    # Filter for categories of interest
    # df = df[df["class"].isin(["Editing->Code", "Editing->Prose", "YouTube", "Twitter"])]
    # df = df[df["class"].isin(["Editing->Code", "Twitter"])]

    if raw:
        _train_raw(df)
    else:
        _train_features(df)


def train_mne():
    """A version that tries to use MNE as much as possible"""
    load()


def _load(use_cache: bool, since: datetime = None) -> pd.DataFrame:
    datacache = Path(".cache/datacache.df.joblib")

    if use_cache:
        logger.info("Loading data from cache...")

        with datacache.open("rb") as f:
            df = joblib.load(f)
    else:
        logger.info("Loading data...")
        df = load.load_labeled_eeg2(since=since)

        # logger.info("Saving to cache...")
        # with datacache.open("wb") as f:
        #     joblib.dump(df, f)

    logger.info("Preprocessing...")
    df = _preprocess(df)

    class_count = {k: v for k, v in dict(df["class"].value_counts()).items() if v}
    logger.info(f"Class count: {class_count}")

    return df


def _train_raw(df):
    """Train a classifier on raw EEG data"""
    X, y = signal_ndarray(df)
    # print(X, y)

    # Fixes non-convergence for binary classification
    dual = set(y) == 2

    clfs: Dict[str, Pipeline] = {
        # These four are from https://neurotechx.github.io/eeg-notebooks/auto_examples/visual_ssvep/02r__ssvep_decoding.html
        "CSP + Cov + TS": make_pipeline(
            Covariances(),
            CSP(4, log=False),
            TangentSpace(),
            LogisticRegression(dual=dual),
        ),
        "Cov + TS": make_pipeline(
            Covariances(), TangentSpace(), LogisticRegression(dual=dual)
        ),
        # Performs meh
        # "CSP + RegLDA": make_pipeline(
        #     Covariances(), CSP(4), LDA(shrinkage="auto", solver="eigen")
        # ),
        # Performs badly
        # "Cov + MDM": make_pipeline(Covariances(), MDM()),
    }

    for name, clf in clfs.items():
        logger.info(f"===== Training with {name} =====")
        _train(X, y, clf)


def _train_features(df: pd.DataFrame):
    """Train a classifier using features"""
    logger.info("Computing features...")
    df = features.compute_features(df)
    X, y = df_to_vectors(df)

    clfs_feat = {
        "SVM": make_pipeline(StandardScaler(), LinearSVC()),
        "randomforest": make_pipeline(RandomForestClassifier(n_estimators=10)),
        # "xgb": make_pipeline(xgb.XGBClassifier(objective="multi:softmax")),
        # "mpl": Pipeline(
        #     steps=[
        #         ("scaler", StandardScaler()),
        #         ("mlp_ann", MLPClassifier(hidden_layer_sizes=(1275, 637))),
        #     ]
        # ),
    }

    for name, clf in clfs_feat.items():
        logger.info(f"===== Training with {name} =====")
        _train(X, y, clf)


def signal_ndarray(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts the raw data to a matrix of shape (n_trials, n_channels, n_samples),
    which is the format required by pyriemann Covariances etc.
    """
    # TODO: Check that array is filled
    n_trials = df.shape[0]
    n_channels = 4
    n_samples = 250 * 5  # sampling freq * min_duration
    X = np.zeros((n_trials, n_channels, n_samples))
    y = np.empty((n_trials))

    catmap = dict(((cls, i) for i, cls in enumerate(df["class"].cat.categories)))
    pprint(catmap)

    i_t = 0
    for _, trial in df.reindex().iterrows():
        n_samples_t = len(trial["raw_data"])
        if n_samples_t > 2 * n_samples:
            logger.warning(
                f"sample count ({n_samples_t}) is significantly greater than ndarray size ({n_samples})"
            )
        if n_samples_t < n_samples:
            logger.warning(
                f"not enough samples ({n_samples_t}) for ndarray size ({n_samples})"
            )
            continue

        for i_s, sample in list(enumerate(trial["raw_data"]))[:n_samples]:
            for i_c, channel in enumerate(sample[1:]):
                X[i_t, i_c, i_s] = channel
        y[i_t] = catmap[trial["class"]]
        i_t += 1

    # Remove skipped trials
    X, y = X[:i_t, :, :], y[:i_t]
    # logger.info(X.shape)
    # logger.info(y.shape)

    return X, y


def _remove_rare(
    df: pd.DataFrame,
    col: str,
    threshold_perc: Optional[float] = None,
    threshold_count: Optional[int] = None,
) -> pd.DataFrame:
    """
    Removes rows with rare categories.

    based on: https://stackoverflow.com/a/31502730/965332
    """
    logger.info(
        f"Removing rare classes... (perc: {threshold_perc}, count: {threshold_count})"
    )
    if threshold_count is not None:
        counts = df[col].value_counts()
        print(counts)
        df = df.loc[df[col].isin(counts[counts > threshold_count].index), :]
    elif threshold_perc is not None:
        counts = df[col].value_counts(normalize=True)
        df = df.loc[df[col].isin(counts[counts > threshold_perc].index), :]
    else:
        raise ValueError
    return df


def classdistribution(df):
    plt.figure(figsize=(12, 5))
    sns.countplot(x=df["class"], color="mediumseagreen")
    plt.title("Class distribution", fontsize=16)
    plt.ylabel("Class Counts", fontsize=16)
    plt.xlabel("Class Label", fontsize=16)
    plt.xticks(rotation="vertical")
    plt.show()


def _train(X, y, clf):
    # Split into train and test
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, test_size=0.3
    )

    logger.info("Training...")

    clf.fit(X_train, y_train)
    logger.info(f"Test score: {clf.score(X_test, y_test)}")

    y_pred = clf.predict(X_test)
    perf = _performance(y_test, y_pred)
    logger.info(perf)
    _save_best_model(clf, perf)

    cross_val(clf, X, y, 3)


def _performance(y_test, y_pred) -> dict:
    precision, recall, fbeta, support = sklearn.metrics.precision_recall_fscore_support(
        y_test, y_pred
    )
    bac = sklearn.metrics.balanced_accuracy_score(y_test, y_pred)
    confusion_matrix = sklearn.metrics.confusion_matrix(y_test, y_pred)
    return {
        "precision": precision.mean(),
        "recall": recall.mean(),
        "fbeta": fbeta.mean(),
        "support": support,
        "bac": bac,
        "confusion_matrix": confusion_matrix,
    }


MODEL = load.cachedir / Path("model.clf")
MODEL_PERF = load.cachedir / Path("model.performance.dict")


def _save_best_model(clf, perf: dict):
    pprint(clf)
    pprint(perf)
    if MODEL_PERF.exists():
        saved_model_perf = joblib.load(MODEL_PERF)
        print("Saved model perf:")
        pprint(saved_model_perf)
        # FIXME: Better criterion
        if perf["bac"] > saved_model_perf["bac"]:
            logger.info("Beat best model! Saving.")
        elif perf["support"].size != saved_model_perf["support"].size:
            logger.info(
                "Training with different number of classes from best model, saving."
            )
        else:
            logger.info("Didn't beat best model, not saving.")
            return
    else:
        logger.info("No model found, saving.")

    joblib.dump(clf, MODEL)
    joblib.dump(perf, MODEL_PERF)


def _load_best_model():
    logger.info("Loading model...")
    return joblib.load(MODEL)


def _load_or_train():
    """Will load any found model, if any, otherwise trains a model."""
    if not MODEL.exists():
        logger.info("No model found, training...")
        train(use_cache=False, raw=True)
    return _load_best_model()


@main.command()
@click.option("--use-cache", is_flag=True)
def predict(use_cache) -> None:
    """Predict the class of a EEG signal"""
    # TODO: Support predicting without labels (modify signal_ndarray)
    clf = _load_or_train()

    since = datetime(2021, 2, 20, tzinfo=timezone.utc)
    df = _load(use_cache, since=since)
    X, y = signal_ndarray(df)
    try:
        print(X.size)
    except Exception as e:
        print(e)

    # TODO: read sample, predict class
    y_pred = clf.predict(X)
    perf = _performance(y, y_pred)
    pprint(perf)


@main.command()
def predict_realtime() -> None:
    """Predict in real time from LSL stream"""
    from eegwatch.devices.base import EEGDevice

    clf = _load_or_train()
    device = EEGDevice.create(device_name="museS")

    X = device._read_buffer()
    n_channels = 4
    X = np.zeros((n_trials, n_channels, n_samples))

    print(X)
    print(clf.predict(X))


def pca(X, y):
    scaler = StandardScaler()
    scaled_df = scaler.fit_transform(X)
    pca = PCA(n_components=20)
    pca_vectors = pca.fit_transform(scaled_df)
    for index, var in enumerate(pca.explained_variance_ratio_):
        logger.info(f"Explained Variance ratio by PC {index + 1}: {var}")

    plt.figure(figsize=(25, 8))
    sns.scatterplot(x=pca_vectors[:, 0], y=pca_vectors[:, 1], hue=y)
    plt.title("Principal Components vs Class distribution", fontsize=16)
    plt.ylabel("Principal Component 2", fontsize=16)
    plt.xlabel("Principal Component 1", fontsize=16)
    plt.xticks(rotation="vertical")
    plt.show()


def _preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses dataframe"""
    min_duration = 5
    df = preprocess.split_rows(df, min_duration)
    df = clean(df)
    df = _remove_rare(df, "class", threshold_perc=0.02)
    return df


@main.command()
def clear():
    """Clears the cache"""
    load.memory.clear()


def df_to_vectors(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    # feature vector
    X = np.array(
        [
            [band for channel in row for band in channel[1:]]
            for row in df["bandpower"].values.tolist()
        ]
    )

    # Map of categories to codes
    catmap = dict(enumerate(df["class"].cat.categories))
    logger.info(f"Classes: {catmap}")

    # label vector
    y = np.array(df["class"].cat.codes)
    # print(y)

    return X, y


def cross_val(clf, X, y, n):
    scores = sklearn.model_selection.cross_val_score(clf, X, y, cv=n)
    logger.info(f"CV score: {scores.mean()}")
    # logger.info(sklearn.model_selection.cross_val_predict(clf, X, y, cv=n))
    # logger.info(sklearn.model_selection.cross_validate(clf, X, y, cv=n))
