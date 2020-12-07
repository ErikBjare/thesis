"""
Based on https://github.com/NeuroTechX/eeg-notebooks/blob/926a63c7f4285ebda72f80a907a552ccf030961e/eegnb/devices/eeg.py

Abstraction for the various supported EEG devices.
    1. Determine which backend to use for the board.
    2.
"""

# TODO: Merge back upstream once https://github.com/NeuroTechX/eeg-notebooks/issues/19 is resolved

import sys
import time
import logging
from time import sleep
from multiprocessing import Process
from typing import List, Tuple

import numpy as np
import pandas as pd

from brainflow import BoardShim, BoardIds, BrainFlowInputParams
import muselsl
from pylsl import StreamInfo, StreamOutlet

logger = logging.getLogger(__name__)

# list of muse devices
muse_devices = [
    "muse2016",
    "muse2",
    "museS",
]

# list of brainflow devices
brainflow_devices = [
    "ganglion",
    "ganglion_wifi",
    "cyton",
    "cyton_wifi",
    "cyton_daisy",
    "cyton_daisy_wifi",
    "brainbit",
    "unicorn",
    "synthetic",
    "brainbit",
    "notion1",
    "notion2",
]

all_devices = muse_devices + brainflow_devices


class EEG:
    def __init__(
        self,
        device: str,
        serial_port=None,
        serial_num=None,
        mac_addr=None,
        other=None,
        ip_addr=None,
    ) -> None:
        """ The initialization function takes the name of the EEG device and determines whether or not
        the device belongs to the Muse or Brainflow families and initializes the appropriate backend.
        Parameters:
            device (str): name of eeg device used for reading data.
        """
        # determine if board uses brainflow or muselsl backend
        self.device_name = device
        self.serial_num = serial_num
        self.serial_port = serial_port
        self.mac_address = mac_addr
        self.ip_addr = ip_addr
        self.other = other
        self.backend = self._get_backend(self.device_name)
        self.initialize_backend()

    def initialize_backend(self):
        if self.backend == "brainflow":
            self._init_brainflow()
        elif self.backend == "muselsl":
            self._init_muselsl()

    def _get_backend(self, device_name: str):
        if device_name in brainflow_devices:
            return "brainflow"
        elif device_name in muse_devices:
            return "muselsl"

    #####################
    #   MUSE functions  #
    #####################
    def _init_muselsl(self):
        # Currently there's nothing we need to do here. However keeping the
        # option open to add things with this init method.
        pass

    def _start_muse(self, duration: float):
        sources = ["EEG"]  # + ["PPG", "ACC", "GYRO"]

        if sys.platform in ["linux", "linux2", "darwin"]:
            # Look for muses
            self.muses = muselsl.list_muses()
            if not self.muses:
                raise Exception("No Muses found")
            # self.muse = muses[0]

            def stream():
                muselsl.stream(
                    self.muses[0]["address"],
                    ppg_enabled="PPG" in sources,
                    acc_enabled="ACC" in sources,
                    gyro_enabled="GYRO" in sources,
                )

            # Start streaming process
            self.stream_process = Process(target=stream, daemon=True)
            self.stream_process.start()

        # Create markers stream outlet
        self.muse_StreamInfo = StreamInfo(
            "Markers", "Markers", 1, 0, "int32", "myuidw43536"
        )
        self.muse_StreamOutlet = StreamOutlet(self.muse_StreamInfo)

        def record(data_source="EEG"):
            filename = self.save_fn
            if data_source != "EEG":
                # TODO: Put the source identifier earlier in the filename (before .csv)
                filename += f".{data_source}"
            muselsl.record(
                duration=duration, filename=filename, data_source=data_source
            )

        # Start a background process that will stream data from the first available Muse
        for source in sources:
            logger.info(
                "Starting background recording process, will save to file: %s"
                % self.save_fn
            )
            self.recording = Process(target=lambda: record(source), daemon=True)
            self.recording.start()

        time.sleep(5)

        self.push_sample([99], timestamp=time.time())

    def _stop_muse(self):
        pass

    def _muse_push_sample(self, marker, timestamp):
        self.muse_StreamOutlet.push_sample(marker, timestamp)

    ##########################
    #   BrainFlow functions  #
    ##########################
    def _init_brainflow(self):
        """ This function initializes the brainflow backend based on the input device name. It calls
        a utility function to determine the appropriate USB port to use based on the current operating system.
        Additionally, the system allows for passing a serial number in the case that they want to use either
        the BraintBit or the Unicorn EEG devices from the brainflow family.
        Parameters:
             serial_num (str or int): serial number for either the BrainBit or Unicorn devices.
        """
        from eegnb.devices.utils import get_openbci_usb

        # Initialize brainflow parameters
        self.brainflow_params = BrainFlowInputParams()

        if self.device_name == "ganglion":
            self.brainflow_id = BoardIds.GANGLION_BOARD.value
            if self.serial_port is None:
                self.brainflow_params.serial_port = get_openbci_usb()
            # set mac address parameter in case
            if self.mac_address is None:
                logger.info(
                    "No MAC address provided, attempting to connect without one"
                )
            else:
                self.brainflow_params.mac_address = self.mac_address

        elif self.device_name == "ganglion_wifi":
            self.brainflow_id = BoardIds.GANGLION_WIFI_BOARD.value
            if self.ip_addr is not None:
                self.brainflow_params.ip_address = self.ip_addr

        elif self.device_name == "cyton":
            self.brainflow_id = BoardIds.CYTON_BOARD.value
            if self.serial_port is None:
                self.brainflow_params.serial_port = get_openbci_usb()

        elif self.device_name == "cyton_wifi":
            self.brainflow_id = BoardIds.CYTON_WIFI_BOARD.value
            if self.ip_addr is not None:
                self.brainflow_params.ip_address = self.ip_addr

        elif self.device_name == "cyton_daisy":
            self.brainflow_id = BoardIds.CYTON_DAISY_BOARD.value
            if self.serial_port is None:
                self.brainflow_params.serial_port = get_openbci_usb()

        elif self.device_name == "cyton_daisy_wifi":
            self.brainflow_id = BoardIds.CYTON_DAISY_WIFI_BOARD.value
            if self.ip_addr is not None:
                self.brainflow_params.ip_address = self.ip_addr

        elif self.device_name == "brainbit":
            self.brainflow_id = BoardIds.BRAINBIT_BOARD.value

        elif self.device_name == "unicorn":
            self.brainflow_id = BoardIds.UNICORN_BOARD.value

        elif self.device_name == "callibri_eeg":
            self.brainflow_id = BoardIds.CALLIBRI_EEG_BOARD.value
            if self.other:
                self.brainflow_params.other_info = str(self.other)

        elif self.device_name == "notion1":
            self.brainflow_id = BoardIds.NOTION_1_BOARD.value

        elif self.device_name == "notion2":
            self.brainflow_id = BoardIds.NOTION_2_BOARD.value

        elif self.device_name == "synthetic":
            self.brainflow_id = BoardIds.SYNTHETIC_BOARD.value

        # some devices allow for an optional serial number parameter for better connection
        if self.serial_num:
            serial_num = str(self.serial_num)
            self.brainflow_params.serial_number = serial_num

        if self.serial_port:
            serial_port = str(self.serial_port)
            self.brainflow_params.serial_port = serial_port

        # Initialize board_shim
        self.sfreq = BoardShim.get_sampling_rate(self.brainflow_id)
        self.board = BoardShim(self.brainflow_id, self.brainflow_params)
        self.board.prepare_session()

    def _start_brainflow(self):
        self.board.start_stream()

    def _stop_brainflow(self):
        """This functions kills the brainflow backend and saves the data to a CSV file."""
        from eegnb.devices.utils import create_stim_array

        # Collect session data and kill session
        data = self.board.get_board_data()  # will clear board buffer
        self.board.stop_stream()
        self.board.release_session()

        # transform data for saving
        data = data.T  # transpose data
        print(data)

        # get the channel names for EEG data
        if self.brainflow_id == BoardIds.GANGLION_BOARD.value:
            # if a ganglion is used, use recommended default EEG channel names
            ch_names = ["fp1", "fp2", "tp7", "tp8"]
        else:
            # otherwise select eeg channel names via brainflow API
            ch_names = BoardShim.get_eeg_names(self.brainflow_id)

        # pull EEG channel data via brainflow API
        eeg_data = data[:, BoardShim.get_eeg_channels(self.brainflow_id)]
        timestamps = data[:, BoardShim.get_timestamp_channel(self.brainflow_id)]

        # Create a column for the stimuli to append to the EEG data
        stim_array = create_stim_array(timestamps, self.markers)
        timestamps = timestamps[
            ..., None
        ]  # Add an additional dimension so that shapes match
        total_data = np.append(timestamps, eeg_data, 1)
        total_data = np.append(
            total_data, stim_array, 1
        )  # Append the stim array to data.

        # Subtract five seconds of settling time from beginning
        # total_data = total_data[5 * self.sfreq :]
        data_df = pd.DataFrame(total_data, columns=["timestamps"] + ch_names + ["stim"])
        data_df.to_csv(self.save_fn, index=False)

    def _brainflow_push_sample(self, marker):
        last_timestamp = self.board.get_current_board_data(1)[-1][0]
        self.markers.append((marker, last_timestamp))

    def start(self, fn: str, duration=None):
        """ Starts the EEG device based on the defined backend.
        Parameters:
            fn (str): name of the file to save the sessions data to.
        """
        if fn:
            self.save_fn = fn

        def record():
            sleep(duration)
            self._stop_brainflow()

        if self.backend == "brainflow":  # Start brainflow backend
            self._start_brainflow()
            self.markers: List[Tuple[int, float]] = []
            if duration:
                logger.info(
                    "Starting background recording process, will save to file: %s"
                    % self.save_fn
                )
                self.recording = Process(target=lambda: record())
                self.recording.start()
        elif self.backend == "muselsl":
            self._start_muse(duration)

    def push_sample(self, marker: List[int], timestamp: float):
        """ Universal method for pushing a marker and its timestamp to store alongside the EEG data.
        Parameters:
            marker (int): marker number for the stimuli being presented.
            timestamp (float): timestamp of stimulus onset from time.time() function.
        """
        if self.backend == "brainflow":
            self._brainflow_push_sample(marker=marker)
        elif self.backend == "muselsl":
            self._muse_push_sample(marker=marker, timestamp=timestamp)

    def stop(self):
        if self.backend == "brainflow":
            self._stop_brainflow()
        elif self.backend == "muselsl":
            pass