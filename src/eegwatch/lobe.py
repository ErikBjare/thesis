"""Service that collects EEG data"""

from eegnb.devices.eeg import EEG


if __name__ == "__main__":
    eeg = EEG("muse2")
    eeg.start()
