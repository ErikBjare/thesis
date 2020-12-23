import sys
import logging
from datetime import datetime, timezone
from time import time, sleep
from multiprocessing import Process
from typing import List

import muselsl
import pylsl

from .base import EEGDevice, _check_samples

logger = logging.getLogger(__name__)


class MuseDevice(EEGDevice):
    # list of muse devices
    devices = [
        "muse2016",
        "muse2",
        "museS",
    ]

    def __init__(self, device_name: str):
        EEGDevice.__init__(self, device_name)

    def start(self, filename: str = None, duration=None):
        """
        Starts the EEG device.

        Parameters:
            filename (str): name of the file to save the sessions data to.
        """
        if filename:
            self.save_fn = filename

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
        self.muse_StreamInfo = pylsl.StreamInfo(
            "Markers", "Markers", 1, 0, "int32", "myuidw43536"
        )
        self.muse_StreamOutlet = pylsl.StreamOutlet(self.muse_StreamInfo)

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

        # FIXME: This shouldn't be here, unnecessarily blocks the thread
        sleep(5)

        # FIXME: What's the purpose of this?
        self.push_sample([99], timestamp=time())

    def stop(self):
        pass

    def push_sample(self, marker: List[int], timestamp: float):
        self.muse_StreamOutlet.push_sample(marker, timestamp)

    def check(self) -> List[str]:
        from eegwatch.lslutils import _get_inlets

        inlets = _get_inlets()
        offset = datetime.now(tz=timezone.utc).timestamp() - pylsl.local_clock()

        def local_clock_to_timestamp(local_clock):
            return local_clock + offset

        def update():
            # Read the last 5s of data from the inlet. Use a timeout of 0.0 so we don't block GUI interaction.
            mintime = local_clock_to_timestamp(pylsl.local_clock()) - 5
            # call pull_and_plot for each inlet.
            # Special handling of inlet types (markers, continuous data) is done in
            # the different inlet classes.
            for inlet in inlets:
                inlet.pull_and_plot(mintime, None)

        _inlets = [inlet for inlet in inlets if inlet.buffer.any()]  # type: ignore
        if not _inlets:
            raise Exception("No inlets found")
        inlet = _inlets[0]
        # print(inlet.inlet)
        # print(inlet.buffer)

        checked = _check_samples(inlet.buffer, channels=["TP9", "AF7", "AF8", "TP10"])  # type: ignore
        bads = [ch for ch, ok in checked.items() if not ok]
        return bads
