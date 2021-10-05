import logging
import itertools
from datetime import datetime, timedelta
from typing import Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def split_rows(df: pd.DataFrame, min_duration: int) -> pd.DataFrame:
    """
    Splits variable-duration rows (epochs) into shorter segments (windows) no shorter than ``min_duration``.

    Note that this might not be suitable for all types of analysis, especially not
    those with a distinct start of stimuli (as opposed to continuous stimuli), like controlled experiments.
    """
    df_orig = df
    while True:
        df_new = _split_rows(df, min_duration)
        if len(df_new.index) == len(df.index):
            break
        df = df_new
    logger.info(f"Split {len(df_orig.index)} epochs into {len(df.index)} windows")
    return df


def _split_rows(df: pd.DataFrame, min_duration: int) -> pd.DataFrame:
    """Split long rows into shorter ones"""
    rows = [(i, _split_row(r, min_duration)) for i, r in df.iterrows()]
    rows_split = [(i, r) for i, r in rows if r is not None]

    # Drop rows that could be split
    df = df.drop(i for i, _ in rows_split)

    # Append split rows
    df_split = pd.DataFrame(itertools.chain(*[r for i, r in rows_split]))
    df = df.append(df_split, ignore_index=True)

    # Not sure why this gets unset in the first place...
    df["class"] = df["class"].astype("category")

    return df


def _split_row(
    row: pd.Series, min_duration: int
) -> Optional[Tuple[pd.Series, pd.Series]]:
    """Splits rows longer than ``min_duration``, else returns None"""
    td = timedelta(seconds=min_duration)
    splitpoint = row["start"] + td
    if splitpoint > row["stop"] - td:
        # Too short to split
        return None

    r1_data = [t for t in row["raw_data"] if t[0] < splitpoint]
    r2_data = [t for t in row["raw_data"] if t[0] > splitpoint]

    r1 = row.copy()
    r1.stop = splitpoint
    r1.raw_data = r1_data

    r2 = row.copy()
    r2.start = splitpoint
    r2.raw_data = r2_data

    return (r1, r2)


def test_split_row():
    df = pd.DataFrame(
        [
            {
                "start": datetime(2020, 1, 1, 12, 0),
                "stop": datetime(2020, 1, 1, 12, 1),
                "class": "test",
                "raw_data": [
                    (datetime(2020, 1, 1, 12, 0, 15), 0.23, 0.13),
                    (datetime(2020, 1, 1, 12, 0, 45), 0.44, 0.07),
                ],
            }
        ],
    )
    df = pd.DataFrame(_split_row(df.iloc[0], 30))
    assert len(df.iloc[0]["raw_data"]) == 1
    assert df.iloc[0]["stop"] == datetime(2020, 1, 1, 12, 0, 30)
    assert df.iloc[1]["start"] == datetime(2020, 1, 1, 12, 0, 30)
