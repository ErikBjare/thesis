import random
import time

from pylsl import StreamInfo, StreamOutlet


def mousedelta():
    # TODO: Add mouse delta as a marker
    # TODO: Add mouse clicks as a marker
    # TODO: Add keyboard presses as a marker

    channels = 1
    uid = "eegwatchmousedelta"
    info = StreamInfo("MyMarkerStream", "Markers", channels, 0, "string", uid)
    outlet = StreamOutlet(info)

    # TODO


def mouseclicks():
    # TODO: Add mouse clicks as a marker

    channels = 1
    uid = "myuidw43536"
    info = StreamInfo("MyMarkerStream", "Markers", channels, 0, "string", uid)
    outlet = StreamOutlet(info)

    # TODO


def keypresses():
    # TODO: Add keyboard presses as a marker

    channels = 1
    uid = "myuidw43536"
    info = StreamInfo("MyMarkerStream", "Markers", channels, 0, "string", uid)
    outlet = StreamOutlet(info)

    # TODO


def marker_str():
    """
    Example with string markers: https://github.com/labstreaminglayer/liblsl-Python/blob/master/pylsl/examples/SendStringMarkers.py
    """
    # TODO: Add current window title as a marker

    channels = 1
    uid = "myuidw43536"
    info = StreamInfo("MyMarkerStream", "Markers", channels, 0, "string", uid)
    outlet = StreamOutlet(info)

    print("now sending markers...")
    markernames = ["Test", "Blah", "Marker", "XXX", "Testtest", "Test-1-2-3"]
    while True:
        # pick a sample to send an wait for a bit
        outlet.push_sample([random.choice(markernames)], timestamp=0.0)
        time.sleep(random.random() * 3)


if __name__ == "__main__":
    marker_str()
