import logging
import itertools

import pandas as pd
import numpy as np
import yasa

from eegwatch.bids import raw_to_mne

logger = logging.getLogger(__name__)


def compute_features(df: pd.DataFrame, ratios: bool = False) -> pd.DataFrame:
    logger.info("Computing features...")
    df = bandpower(df, ratios=ratios)
    logger.info("Done computing features!")
    return df


BAND_NAMES = ["Delta", "Theta", "Alpha", "Sigma", "Beta", "Gamma"]


def bandpower(df: pd.DataFrame, ratios=False) -> pd.DataFrame:
    """
    Computes bandpower features using the bands in BAND_NAMES, and
    optionally computes all permutations of ratios between the bands.
    """
    df.loc[:, "bandpower"] = [[] for _ in range(len(df))]
    for i, row in df.iterrows():
        # Handle differences in input data (may be with or without timestamps)
        if len(row["raw_data"][0]) == 5:
            # Timestamps in data
            timestamps, *ch = zip(*row["raw_data"])
        elif len(row["raw_data"][0]) == 4:
            # No timestamps in data
            ch = list(zip(*row["raw_data"]))
        else:
            raise ValueError

        data = np.array(ch).T
        rawarray = raw_to_mne(data)
        df_power = yasa.bandpower(rawarray)
        df_power = df_power.reset_index()[["Chan", *BAND_NAMES]]

        # Compute band ratios
        if ratios:
            for b1, b2 in itertools.permutations(BAND_NAMES, 2):
                df_power[f"{b1}/{b2}"] = df_power[b1] / df_power[b2]

        df.at[i, "bandpower"] = df_power.values.tolist()

    return df
