import os
import logging
from pathlib import Path

import joblib
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

srcpath = Path(os.path.realpath(__file__))
rootdir = srcpath.parent.parent.parent
datadir = rootdir / "data"
cachedir = rootdir / ".cache"

memory = joblib.Memory(location=cachedir)

CHANNELS_MUSE = ["TP9", "AF7", "AF8", "TP10"]


def load_labels() -> pd.DataFrame:
    path_awdata = datadir / "aw"
    path_labels_prod = path_awdata / "labels.csv"
    path_labels_test = path_awdata / "labels.test.csv"

    # Should always exist
    assert path_labels_test.exists()

    if path_labels_prod.exists():
        path_labels = path_labels_prod
    else:
        logger.warning("Using testing data")
        path_labels = path_labels_test

    with path_labels.open("r") as f:
        df = pd.read_csv(f)

    return df


def test_load_labels():
    assert not load_labels().empty


def load_eeg() -> pd.DataFrame:
    # TODO: Parametrize for different devices/sources
    # TODO: Parametrize for subject/session?
    musedir = datadir / "eeg" / "muse"

    # Load all files
    # TODO: Might be excessive, can be a lot of data which might slow things significantly
    fns = musedir.glob("subject*/session*/*.csv")

    # Only committed example EEG data (the only file available for tests in CI)
    # Matches up with labels in aw/labels.test.csv
    # fns = musedir.glob("subject*/session*/recording_2020-11-01-13.16.04.csv")

    dfs = []
    for fn in fns:
        with fn.open("r") as f:
            df = pd.read_csv(f).rename(columns={"timestamps": "timestamp"})
            # FIXME: Which timezone is used by LSL/muse-lsl?
            #        We'll assume UTC for now: https://github.com/ErikBjare/thesis/issues/16
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
            dfs.append(df)

    return pd.concat(dfs)


def test_load_eeg():
    assert not load_eeg().empty


def _label_data(df_eeg: pd.DataFrame, df_labels: pd.DataFrame) -> pd.DataFrame:
    df_eeg["class"] = np.nan

    for i, row in df_labels.iterrows():
        idxs = (row["start"] < df_eeg["timestamp"]) & (
            df_eeg["timestamp"] < row["stop"]
        )
        df_eeg.loc[idxs, "class"] = row["class"]

    df_eeg["class"] = df_eeg["class"].astype("category")

    return df_eeg


def load_labeled_eeg() -> pd.DataFrame:
    df_eeg = load_eeg()
    df_labels = load_labels()
    return _label_data(df_eeg, df_labels)


def test_load_labeled_eeg():
    df = load_labeled_eeg()

    # Check that there is data
    assert not df.empty

    # Check that data has labels
    assert not df["class"].dropna().empty

    # print(set(df["class"]))
    # print(df.dropna(subset=["class"]))


# @memory.cache
def load_labeled_eeg2() -> pd.DataFrame:
    """
    Similar to load_labeled_eeg, but gives one row per task-epoch, with EEG data as cell-vector.

    This is similarly structured to datasets like the Berkeley Synchronized Brainwave Dataset.
    """
    channels = CHANNELS_MUSE

    df_eeg = load_eeg()
    df_labels = load_labels()
    df_labels["raw_data"] = [() for _ in range(len(df_labels))]

    for i, row in df_labels.iterrows():
        # Get data within range of label
        idxs = (row["start"] < df_eeg["timestamp"]) & (
            df_eeg["timestamp"] < row["stop"]
        )
        raw_data = df_eeg.loc[idxs, ["timestamp", *channels]]

        # Convert to list of (timestamp, *channels) tuples
        raw_data = [
            (r["timestamp"],) + tuple(r[ch] for ch in channels)
            for i, r in raw_data.iterrows()
        ]
        df_labels.at[i, "raw_data"] = raw_data

    # Not sure where this column is coming from
    # df_labels = df_labels.drop(columns="Unnamed: 0")

    # Drop rows without data
    df_labels = df_labels[df_labels["raw_data"].map(len) > 0]

    return df_labels


def test_load_labeled_eeg2():
    df = load_labeled_eeg2()

    # Check that there is data
    assert not df.empty
