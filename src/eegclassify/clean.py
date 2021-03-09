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
    thres = 200
    return all(_check_samples(buffer, channels, max_uv_abs=thres).values())


def _clean_signal_quality(df: pd.DataFrame) -> pd.DataFrame:
    bads = []
    for i, row in df.iterrows():
        if not _check_row_signal_quality(row):
            bads.append(i)

    logger.warning(f"Dropping {len(bads)} bad rows due to bad signal quality")
    return df.drop(bads)
