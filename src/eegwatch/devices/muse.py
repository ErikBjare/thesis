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

backend = "bleak"


def stream(address, sources):
    muselsl.stream(
        address,
        backend="bleak",
        ppg_enabled="PPG" in sources,
        acc_enabled="ACC" in sources,
        gyro_enabled="GYRO" in sources,
    )


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
        sources = ["EEG"]  # + ["PPG", "ACC", "GYRO"]
        if not duration:
            duration = 300

        # Not sure why we only do this on *nix
        # Makes it seem like streaming is only supported on *nix?
        if sys.platform in ["linux", "linux2", "darwin"]:
            # Look for muses
            muses = muselsl.list_muses(backend="bleak")
            # FIXME: fix upstream
            muses = [m for m in muses if m["name"].startswith("Muse")]
            if not muses:
                raise Exception("No Muses found")
            # self.muse = muses[0]

            # Start streaming process
            self.stream_process = Process(
                target=stream, args=(muses[0]["address"], sources), daemon=True
            )
            self.stream_process.start()

        # Create markers stream outlet
        self.marker_outlet = pylsl.StreamOutlet(
            pylsl.StreamInfo("Markers", "Markers", 1, 0, "int32", "myuidw43536")
        )

        def record(data_source="EEG"):
            muselsl.record(
                duration=duration, filename=filename, data_source=data_source
            )

        # Start a background process that will stream data from the first available Muse
        for source in sources:
            logger.info("Starting background recording process")
            self.rec_process = Process(target=lambda: record(source), daemon=True)
            self.rec_process.start()

        # FIXME: What's the purpose of this?
        self.push_sample([99], timestamp=time())

    def stop(self):
        pass

    def push_sample(self, marker: List[int], timestamp: float):
        self.marker_outlet.push_sample(marker, timestamp)

    def check(self) -> List[str]:
        from eegwatch.lslutils import _get_inlets

        inlets = _get_inlets(verbose=False)

        for i in range(5):
            for inlet in inlets:
                inlet.pull(timeout=0.5)  # type: ignore
            inlets = [inlet for inlet in inlets if inlet.buffer.any()]  # type: ignore
            if inlets:
                break
            else:
                logger.info("No inlets with data, trying again in a second...")
                sleep(1)

        if not inlets:
            raise Exception("No inlets found")

        inlet = inlets[0]
        checked = _check_samples(inlet.buffer, channels=["TP9", "AF7", "AF8", "TP10"])  # type: ignore
        bads = [ch for ch, ok in checked.items() if not ok]
        return bads
