from pathlib import Path

from time import sleep

import matplotlib.pyplot as plt
import numpy as np
import mne
import mne_bids

from eegwatch import data_dir


def csv_to_fif(path: Path):
    """Load a CSV created by muse-lsl and save as BIDS"""
    path_out = data_dir / "test.raw.fif"

    data = np.loadtxt(path, delimiter=",", skiprows=1)
    start = data[0][0]
    print(f"started at {start}")
    # Drop first and last col (timestamp and Right AUX)
    data = np.delete(data, 5, axis=1)
    data = np.delete(data, 0, axis=1)

    ch_names = ["TP9", "AF7", "AF8", "TP10"]
    sfreq = 250  # The Muse S uses 250Hz

    info = mne.create_info(ch_names, sfreq)
    info["line_freq"] = 50
    raw = mne.io.RawArray(data.T, info)
    raw.set_channel_types({ch: "eeg" for ch in ch_names})

    # print(raw)
    # raw.plot()
    # plt.show()

    include = ["TP9", "AF7", "AF8", "TP10"]
    raw.info["bads"] += []
    picks = mne.pick_types(raw.info, eeg=True, include=include, exclude="bads")

    # https://mne.tools/dev/auto_examples/io/plot_read_and_write_raw_data.html
    raw.save(path_out, tmin=0, tmax=150, picks=picks, overwrite=True)
    return path_out


def fif_to_bids(path: Path):
    # TODO: Load FIF
    raw = mne.io.read_raw_fif(path)

    bids_path = mne_bids.BIDSPath(
        subject="01",
        session="01",
        task="testing",
        acquisition="01",
        run="01",
        root=data_dir / "BIDS",
    )
    mne_bids.write_raw_bids(raw, bids_path, overwrite=True)


if __name__ == "__main__":
    path_fif = csv_to_fif(data_dir / "EEG_recording_2020-09-30-09.18.34.csv")
    path_bids = fif_to_bids(path_fif)
