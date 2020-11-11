import logging

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yasa

from eegwatch.bids import raw_to_mne
from . import load

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO)

    df = load.load_labeled_eeg2()
    print(df)
    raw_data = df.iloc[0]["raw_data"]
    timestamps, *ch = zip(*raw_data)
    # plt.plot(timestamps, ch[0])
    # sns.despine()
    # plt.show()

    # TODO: Refactor
    data = np.array(ch).T
    rawarray = raw_to_mne(data)
    df_power = yasa.bandpower(rawarray)
    print(df_power)
