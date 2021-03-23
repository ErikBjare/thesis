import logging

import pandas as pd
import numpy as np

from eegwatch.devices.base import _check_samples

logger = logging.getLogger(__name__)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    # TODO: Add notch filters for 50Hz and 60Hz?
    df = _clean_short(df)
    df = _clean_duplicate_samples(df)
    df = _clean_signal_quality(df)
    df = _clean_inconsistent_sampling(df)
    return df


def _clean_short(df: pd.DataFrame) -> pd.DataFrame:
    sfreq = 250  # NOTE: Will be different for different devices
    bads = []
    for i, row in df.iterrows():
        samples = len(row["raw_data"])
        seconds = samples / sfreq
        duration = row["stop"] - row["start"]
        if seconds < 0.95 * duration.total_seconds():
            # logger.warning(
            #     f"Bad row found, only had {seconds}s of data out of {duration.total_seconds()}s"
            # )
            bads.append(i)

    logger.warning(f"Dropping {len(bads)} rows due to missing samples")
    return df.drop(bads)


def _clean_duplicate_samples(df: pd.DataFrame) -> pd.DataFrame:
    bads = []
    for i, row in df.iterrows():
        timestamps = [sample[0] for sample in row["raw_data"]]

        # Check for duplicate timestamps
        if len(timestamps) != len(set(timestamps)):
            bads.append(i)

    logger.warning(f"Dropping {len(bads)} bad rows due to duplicate samples")
    return df.drop(bads)


def _clean_inconsistent_sampling(df: pd.DataFrame) -> pd.DataFrame:
    bads = []
    for i, row in df.iterrows():
        timestamps = [sample[0] for sample in row["raw_data"]]

        # Check for consistent sampling
        diffs = np.diff(timestamps)
        if max(diffs) > 2 * min(diffs):
            bads.append(i)

    logger.warning(f"Dropping {len(bads)} bad rows due to inconsistent sampling")
    return df.drop(bads)


def _check_row_signal_quality(s: pd.Series) -> bool:
    """Takes a single row as input, returns true if good signal quality else false"""
    # TODO: Improve quality detection
    buffer = np.array([t[1:] for t in s["raw_data"]])  # strip timestamp
    channels = [str(i) for i in range(buffer.size)]
    thres = 500
    return all(_check_samples(buffer, channels, max_uv_abs=thres).values())


def _row_stats(s: pd.Series):
    buffer = np.array([t[1:] for t in s["raw_data"]])  # strip timestamp
    return {
        "min": np.min(np.abs(buffer), axis=0),
        "max": np.max(np.abs(buffer), axis=0),
        "ok": _check_row_signal_quality(s),
    }


def _clean_signal_quality(df: pd.DataFrame) -> pd.DataFrame:
    bads = []
    for i, row in df.iterrows():
        # print(_row_stats(row))
        if not _check_row_signal_quality(row):
            bads.append(i)

    logger.warning(f"Dropping {len(bads)} bad rows due to bad signal quality")
    return df.drop(bads)


def test_clean_signal_quality():
    df = pd.DataFrame(
        [
            {"raw_data": [[1, 100]]},
            {"raw_data": [[1, 100]]},
            {"raw_data": [[1, 300]]},
        ]
    )
    df_clean = _clean_signal_quality(df)
    print(df_clean)
    assert len(df_clean) == 2
    assert False


def _select_classes(
    df: pd.DataFrame,
    col: str,
    classes: List[str],
) -> pd.DataFrame:
    """
    Removes rows that don't match the selected classes.
    """
    return df.loc[df[col].isin(classes), :]


def test_select_classes():
    df = pd.DataFrame({"class": ["a", "a", "b", "c"]})
    df["class"] = df["class"].astype("category")
    df = _select_classes(df, "class", ["a", "b"])
    assert len(df) == 3


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
    if threshold_perc is None and threshold_count is None:
        raise ValueError

    logger.info(
        f"Removing rare classes... (perc: {threshold_perc}, count: {threshold_count})"
    )
    if threshold_count is not None:
        counts = df[col].value_counts()
        df = df.loc[df[col].isin(counts[counts >= threshold_count].index), :]
    if threshold_perc is not None:
        counts = df[col].value_counts(normalize=True)
        df = df.loc[df[col].isin(counts[counts >= threshold_perc].index), :]

    return df


def test_remove_rare():
    df = pd.DataFrame({"class": ["a"] * 10 + ["b"] * 5 + ["c"] * 3 + ["d"] * 2})
    df["class"] = df["class"].astype("category")

    # Remove single class with percent
    assert len(_remove_rare(df, "class", threshold_perc=0.15)) == 18

    # Remove single class with count
    assert len(_remove_rare(df, "class", threshold_count=3)) == 18

    # Remove one class by count and one by percent
    assert len(_remove_rare(df, "class", threshold_perc=0.2, threshold_count=3)) == 15
