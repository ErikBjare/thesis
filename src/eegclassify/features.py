import logging

import pandas as pd
import numpy as np
import yasa

from eegwatch.bids import raw_to_mne

logger = logging.getLogger(__name__)


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Computing features...")
    df = bandpower(df)
    logger.info("Done computing features!")
    return df


def bandpower(df: pd.DataFrame) -> pd.DataFrame:
    df.loc[:, "bandpower"] = [[] for _ in range(len(df))]
    for i, row in df.iterrows():
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
        df_power = df_power.reset_index()[
            ["Chan", "Delta", "Theta", "Alpha", "Sigma", "Beta", "Gamma"]
        ]
        df.at[i, "bandpower"] = df_power.values.tolist()

    return df
