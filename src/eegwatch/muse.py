"""
Stuff to get EEG and other data from the Muse S.
"""

from typing import Optional
from threading import Thread

# from multiprocessing import Process

from pylsl import StreamInfo, StreamOutlet
from muselsl import stream, list_muses, record, view


def main(address: Optional[str] = None):
    if address is None:
        muses = list_muses()
        assert muses
        address = muses[0]["address"]

    s = Thread(target=_stream, args=(address,))
    s.start()


def _stream(address):
    stream(address, ppg_enabled=True, acc_enabled=True, gyro_enabled=True)

    # Note: Streaming is synchronous, so code here will not execute until after the stream has been closed
    print("Stream has ended")


if __name__ == "__main__":
    main("00:55:DA:B9:27:54")
