import os
import logging
from pathlib import Path
from typing import List
from multiprocessing import Pool
from datetime import datetime, timezone

from tqdm import tqdm

import joblib
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

srcpath = Path(os.path.realpath(__file__))
rootdir = srcpath.parent.parent.parent
datadir = rootdir / "data"
cachedir = rootdir / ".cache"
musedir = datadir / "eeg" / "muse"

musesub0dir = musedir / "subject0000" / "session001"
TEST_EEG_FILES_MUSE = [
    musesub0dir / "recording_2020-09-30-09.18.34.csv",
    musesub0dir / "recording_2020-11-01-13.16.04.csv",
]

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
        df = pd.read_csv(f, parse_dates=["start", "stop"])

    df["class"] = df["class"].astype("category")

    return df


def test_load_labels():
    assert not load_labels().empty


def load_eeg(files: List[Path] = None) -> pd.DataFrame:
    # TODO: Parametrize for different devices/sources
    # TODO: Parametrize for subject/session?

    if not files:
        # Load all files
        files = sorted(list(musedir.glob("subject*/session*/*.csv")))
        files = reversed(files)  # type: ignore

    return _load_eeg(files)


def _read_csv(fn: Path) -> pd.DataFrame:
    with fn.open("r") as f:
        df = pd.read_csv(f).rename(columns={"timestamps": "timestamp"})
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    return df


@memory.cache
def _load_eeg(files: List[Path]) -> pd.DataFrame:
    with Pool(4) as p:
        dfs: List[pd.DataFrame] = p.map(_read_csv, files)

    print("Concatenating...")
    df = pd.concat(dfs)
    print("Concatenated!")
    df = df.drop(columns=["Marker0", "Right AUX"], errors="ignore")
    return df


def test_load_eeg():
    assert not load_eeg(TEST_EEG_FILES_MUSE).empty


def _label_data(df_eeg: pd.DataFrame, df_labels: pd.DataFrame) -> pd.DataFrame:
    df_eeg["class"] = np.nan

    for i, row in df_labels.iterrows():
        idxs = (row["start"] < df_eeg["timestamp"]) & (
            df_eeg["timestamp"] < row["stop"]
        )
        df_eeg.loc[idxs, "class"] = row["class"]

    df_eeg["class"] = df_eeg["class"].astype("category")
    return df_eeg


def load_labeled_eeg(files=None) -> pd.DataFrame:
    """Returns a dataframe with columns: timestamp,*channels,class"""
    df_eeg = load_eeg(files)
    df_labels = load_labels()
    df = _label_data(df_eeg, df_labels)
    df = df.dropna(subset=["class"])
    return df


def test_load_labeled_eeg():
    df = load_labeled_eeg(TEST_EEG_FILES_MUSE)

    # Check that there is data
    assert not df.empty

    # Check that data has labels
    assert not df["class"].dropna().empty

    assert list(df.columns) == ["timestamp", *CHANNELS_MUSE, "class"]


def load_labeled_eeg2(files=None, since: datetime = None) -> pd.DataFrame:
    """
    Similar to load_labeled_eeg, but gives one row per task-epoch, with EEG data as cell-vector.

    This is similarly structured to datasets like the Berkeley Synchronized Brainwave Dataset.
    """
    channels = CHANNELS_MUSE

    df_eeg = load_eeg(files)
    df_labels = load_labels()
    df_labels["raw_data"] = [() for _ in range(len(df_labels))]

    # since = datetime(2021, 2, 1, tzinfo=timezone.utc)
    if since:
        logger.info(f"Truncating all before {since}")
        df_eeg = df_eeg[df_eeg["timestamp"] > since]
        df_labels = df_labels[df_labels["stop"] > since]

    logger.info("Transforming...")
    for i, row in tqdm(df_labels.iterrows(), total=len(df_labels)):
        # Get data within range of label
        idxs = (row["start"] < df_eeg["timestamp"]) & (
            df_eeg["timestamp"] < row["stop"]
        )
        if idxs.empty:
            continue
        raw_data = df_eeg.loc[idxs, ["timestamp", *channels]]

        # Convert to list of (timestamp, *channels) tuples
        raw_data = [
            (r["timestamp"],) + tuple(r[ch] for ch in channels)
            for i, r in raw_data.iterrows()
        ]
        df_labels.at[i, "raw_data"] = raw_data
    logger.info("Transforming done!")

    # Drop rows without data
    df_labels = df_labels[df_labels["raw_data"].map(len) > 0]

    return df_labels


def test_load_labeled_eeg2():
    df = load_labeled_eeg2(TEST_EEG_FILES_MUSE)

    # Check that there is data
    assert not df.empty

    # Check that columns are correct
    assert list(df.columns) == ["start", "stop", "class", "raw_data"]
