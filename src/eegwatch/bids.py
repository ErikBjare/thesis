from pathlib import Path

import pytest
import numpy as np
import mne
import mne_bids

from eegwatch import data_dir

CHANNELS_MUSE = ["TP9", "AF7", "AF8", "TP10"]

PATH_TESTFILE = (
    data_dir
    / "eeg"
    / "muse"
    / "subject0000"
    / "session001"
    / "recording_2020-09-30-09.18.34.csv"
)


def raw_to_mne(data: np.ndarray, first_samp=0) -> mne.io.RawArray:
    ch_names = CHANNELS_MUSE
    sfreq = 250  # The Muse S uses 250Hz

    info = mne.create_info(ch_names, sfreq, ch_types="eeg")
    info["line_freq"] = 50
    raw = mne.io.RawArray(data.T, info, verbose=False)
    # raw.set_channel_types({ch: "eeg" for ch in ch_names})

    # print(raw)
    # raw.plot()
    # plt.show()

    raw.info["bads"] += []
    return raw


def csv_to_mne(path: Path) -> mne.io.RawArray:
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    start = data[0][0]
    print(f"started at {start}")
    # Drop first and last col (timestamp and Right AUX)
    data = np.delete(data, 5, axis=1)
    data = np.delete(data, 0, axis=1)

    return raw_to_mne(data, first_samp=start)


def test_csv_to_mne():
    assert csv_to_mne(PATH_TESTFILE)


def csv_to_fif(path: Path):
    """Load a CSV created by muse-lsl and save as BIDS"""
    path_out = data_dir / "test.raw.fif"

    raw = csv_to_mne(path)

    include = CHANNELS_MUSE
    picks = mne.pick_types(raw.info, eeg=True, include=include, exclude="bads")

    # https://mne.tools/dev/auto_examples/io/plot_read_and_write_raw_data.html
    raw.save(path_out, tmin=0, tmax=150, picks=picks, overwrite=True)
    return path_out


def fif_to_bids(path: Path):
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


@pytest.mark.filterwarnings("ignore:Converting")
def test_csv_to_bids():
    path_fif = csv_to_fif(PATH_TESTFILE)
    path_bids = fif_to_bids(path_fif)
    print(path_bids)
