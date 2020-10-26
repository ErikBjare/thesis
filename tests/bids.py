from eegwatch import data_dir
from eegwatch.bids import csv_to_fif, fif_to_bids


def test_csv_to_bids():
    path_fif = csv_to_fif(data_dir / "muse" / "EEG_recording_2020-09-30-09.18.34.csv")
    path_bids = fif_to_bids(path_fif)
    print(path_bids)
