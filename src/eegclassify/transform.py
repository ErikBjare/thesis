import logging
from pprint import pprint
from typing import Tuple

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


def signal_ndarray(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts the raw data to a matrix of shape (n_trials, n_channels, n_samples),
    which is the format required by pyriemann Covariances etc.
    """
    logger.info("Constructing array...")
    # TODO: Check that array is filled
    n_trials = df.shape[0]
    n_channels = 4
    n_samples = 250 * 5  # sampling freq * min_duration
    X = np.zeros((n_trials, n_channels, n_samples))
    y = np.empty((n_trials))

    catmap = dict(((cls, i) for i, cls in enumerate(df["class"].cat.categories)))
    # pprint(catmap)

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
