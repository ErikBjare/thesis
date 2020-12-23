"""
Based on https://github.com/NeuroTechX/eeg-notebooks/blob/926a63c7f4285ebda72f80a907a552ccf030961e/eegnb/devices/eeg.py

Abstraction for the various supported EEG devices.
"""

# TODO: Merge back upstream once https://github.com/NeuroTechX/eeg-notebooks/issues/19 is resolved

import logging
from typing import List, Dict
from abc import ABCMeta, abstractmethod

import numpy as np


logger = logging.getLogger(__name__)


def _check_samples(buffer: np.ndarray, channels: List[str]) -> Dict[str, bool]:
    # TODO: Better signal quality check
    # TODO: Merge with signal check filter in eegclassify
    chmax = dict(zip(channels, np.max(np.abs(buffer), axis=0),))
    return {ch: maxval < 200 for ch, maxval in chmax.items()}


class EEGDevice(metaclass=ABCMeta):
    def __init__(self, device: str) -> None:
        """
        The initialization function takes the name of the EEG device and initializes the appropriate backend.

        Parameters:
            device (str): name of eeg device used for reading data.
        """
        self.device_name = device

    @classmethod
    def create(cls, device_name: str, *args, **kwargs) -> "EEGDevice":
        from .muse import MuseDevice
        from ._brainflow import BrainflowDevice

        if device_name in BrainflowDevice.devices:
            return BrainflowDevice(device_name)
        elif device_name in MuseDevice.devices:
            return MuseDevice(device_name)
        else:
            raise ValueError(f"Invalid device name: {device_name}")

    @abstractmethod
    def start(self, filename: str = None, duration=None):
        """
        Starts the EEG device based on the defined backend.

        Parameters:
            filename (str): name of the file to save the sessions data to.
        """
        raise NotImplementedError

    @abstractmethod
    def push_sample(self, marker: List[int], timestamp: float):
        """
        Push a marker and its timestamp to store alongside the EEG data.

        Parameters:
            marker (int): marker number for the stimuli being presented.
            timestamp (float): timestamp of stimulus onset from time.time() function.
        """
        raise NotImplementedError

    @abstractmethod
    def stop(self):
        raise NotImplementedError

    @abstractmethod
    def check(self):
        raise NotImplementedError
