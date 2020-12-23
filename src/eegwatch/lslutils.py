# Based on https://github.com/labstreaminglayer/liblsl-Python/blob/master/pylsl/examples/ReceiveAndPlot.py

import math
import logging
from typing import List, Optional

import numpy as np
import pylsl
import pyqtgraph as pg

PLOT_DURATION = 5  # how many seconds of data to show
PULL_INTERVAL = 500  # ms between each pull operation

logger = logging.getLogger(__name__)


def _resolve_streams() -> list:
    logger.info("Finding stream...")
    # Get the Muse EEG stream
    streams = pylsl.resolve_bypred("name='Muse' and type='EEG'", timeout=10)
    if not streams:
        logger.error("No appropriate stream could be found")
        exit(1)
    return streams


def _get_inlets(plt=None) -> List["Inlet"]:
    streams = _resolve_streams()

    inlets: List[Inlet] = []

    # iterate over found streams, creating specialized inlet objects that will
    # handle plotting the data
    for info in streams:
        if info.type() == "Markers":
            if (
                info.nominal_srate() != pylsl.IRREGULAR_RATE
                or info.channel_format() != pylsl.cf_string
            ):
                logger.warning("Invalid marker stream " + info.name())
            logger.info("Adding marker inlet: " + info.name())
            inlets.append(MarkerInlet(info))
        elif (
            info.nominal_srate() != pylsl.IRREGULAR_RATE
            and info.channel_format() != pylsl.cf_string
        ):
            logger.info(f"Adding data inlet '{info.name()}' of type '{info.type()}'")
            inlets.append(DataInlet(info, plt))
        else:
            logger.warning(
                f"Don't know what to do with stream {info.name()} of type {info.type()}"
            )

    return inlets


class Inlet:
    """Base class to represent a plottable inlet"""

    def __init__(self, info: pylsl.StreamInfo):
        # create an inlet and connect it to the outlet we found earlier.
        # max_buflen is set so data older the plot_duration is discarded
        # automatically and we only pull data new enough to show it

        # Also, perform online clock synchronization so all streams are in the
        # same time domain as the local lsl_clock()
        # (see https://labstreaminglayer.readthedocs.io/projects/liblsl/ref/enums.html#_CPPv414proc_clocksync)
        # and dejitter timestamps
        self.inlet = pylsl.StreamInlet(
            info,
            max_buflen=PLOT_DURATION,
            # TODO: This will only work with a newer pylsl than supported by muselsl/eegnb
            # processing_flags=pylsl.proc_clocksync | pylsl.proc_dejitter,
        )
        # store the name and channel count
        self.name = info.name()
        self.channel_count = info.channel_count()

    def pull_and_plot(self, plot_time: float, plt: pg.PlotItem):
        """Pull data from the inlet and add it to the plot.
        :param plot_time: lowest timestamp that's still visible in the plot
        :param plt: the plot the data should be shown on
        """
        # We don't know what to do with a generic inlet, so we skip it.
        pass


class DataInlet(Inlet):
    """A DataInlet represents an inlet with continuous, multi-channel data that
    should be plotted as multiple lines."""

    dtypes = [[], np.float32, np.float64, None, np.int32, np.int16, np.int8, np.int64]

    def __init__(self, info: pylsl.StreamInfo, plt: Optional[pg.PlotItem]):
        super().__init__(info)
        # calculate the size for our buffer, i.e. two times the displayed data
        self.bufsize = (
            2 * math.ceil(info.nominal_srate() * PLOT_DURATION),
            info.channel_count(),
        )
        self.buffer = np.empty(self.bufsize, dtype=self.dtypes[info.channel_format()])
        empty = np.array([])
        # create one curve object for each channel/line that will handle displaying the data
        self.curves = []
        if plt:
            self.curves = [
                pg.PlotCurveItem(x=empty, y=empty, autoDownsample=True)
                for _ in range(self.channel_count)
            ]
            for curve in self.curves:
                plt.addItem(curve)

    def pull_and_plot(self, plot_time, plt):
        # pull the data
        samples, ts = self.inlet.pull_chunk(timeout=0.0, max_samples=self.bufsize[0])
        self.buffer = np.asarray(samples)
        # ts will be empty if no samples were pulled, a list of timestamps otherwise
        if plt and ts:
            ts = np.asarray(ts)
            y = self.buffer[0 : ts.size, :]
            this_x = None
            old_offset = 0
            new_offset = 0
            for ch_ix in range(self.channel_count):
                # we don't pull an entire screen's worth of data, so we have to
                # trim the old data and append the new data to it
                old_x, old_y = self.curves[ch_ix].getData()
                # the timestamps are identical for all channels, so we need to do
                # this calculation only once
                if ch_ix == 0:
                    # find the index of the first sample that's still visible,
                    # i.e. newer than the left border of the plot
                    old_offset = old_x.searchsorted(plot_time)
                    # same for the new data, in case we pulled more data than
                    # can be shown at once
                    new_offset = ts.searchsorted(plot_time)
                    # append new timestamps to the trimmed old timestamps
                    this_x = np.hstack((old_x[old_offset:], ts[new_offset:]))
                # append new data to the trimmed old data
                this_y = np.hstack((old_y[old_offset:], y[new_offset:, ch_ix] - ch_ix))
                # replace the old data
                self.curves[ch_ix].setData(this_x, this_y)


class MarkerInlet(Inlet):
    """A MarkerInlet shows events that happen sporadically as vertical lines"""

    def __init__(self, info: pylsl.StreamInfo):
        super().__init__(info)

    def pull_and_plot(self, plot_time, plt):
        # TODO: purge old markers
        strings, timestamps = self.inlet.pull_chunk(0)
        if timestamps:
            for string, ts in zip(strings, timestamps):
                plt.addItem(
                    pg.InfiniteLine(ts, angle=90, movable=False, label=string[0])
                )
