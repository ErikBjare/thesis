import pandas as pd
import numpy as np

import yasa

from eegwatch.bids import raw_to_mne


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    # How to add to pandas frame?
    df = bandpower(df)
    return df


def bandpower(df: pd.DataFrame) -> pd.DataFrame:
    df.loc[:, "bandpower"] = [[] for _ in range(len(df))]
    for i, row in df.iterrows():
        timestamps, *ch = zip(*row["raw_data"])
        data = np.array(ch).T
        rawarray = raw_to_mne(data)
        df_power = yasa.bandpower(rawarray)
        df_power = df_power.reset_index()[
            ["Chan", "Delta", "Theta", "Alpha", "Sigma", "Beta", "Gamma"]
        ]
        df.at[i, "bandpower"] = df_power.values.tolist()

    return df
