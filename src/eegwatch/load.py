import time

import mne
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds


def load_demo(duration=1) -> mne.io.RawArray:
    """Loads and returns some demo data"""
    BoardShim.enable_dev_board_logger()
    # use synthetic board for demo
    params = BrainFlowInputParams()
    board = BoardShim(BoardIds.SYNTHETIC_BOARD.value, params)
    board.prepare_session()
    board.start_stream()

    print("Waiting for board shim to generate data...")
    time.sleep(duration)
    print("Done!")

    data = board.get_board_data()
    board.stop_stream()
    board.release_session()

    eeg_channels = BoardShim.get_eeg_channels(BoardIds.SYNTHETIC_BOARD.value)
    # Only pick 8 channels
    eeg_channels = eeg_channels[:8]

    eeg_data = data[eeg_channels, :]
    eeg_data = eeg_data / 1000000  # BrainFlow returns uV, convert to V for MNE

    # Creating MNE objects from brainflow data arrays
    ch_types = ["eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg"]
    ch_names = ["T7", "CP5", "FC5", "C3", "C4", "FC6", "CP6", "T8"]
    sfreq = BoardShim.get_sampling_rate(BoardIds.SYNTHETIC_BOARD.value)
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    return mne.io.RawArray(eeg_data, info)


def test_load_demo():
    data = load_demo(0.2)
    assert data
