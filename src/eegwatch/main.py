"""
Basic watcher for continuous EEG recording, with some visualization (soonTM).

eegnb stuff is based on: https://neurotechx.github.io/eeg-notebooks/auto_examples/visual_n170/00x__n170_run_experiment.html#sphx-glr-auto-examples-visual-n170-00x-n170-run-experiment-py
pylsl stuff is based on: https://github.com/labstreaminglayer/liblsl-Python/blob/master/pylsl/examples/ReceiveAndPlot.py
"""

from typing import List
from time import sleep

import logging
import click
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import pylsl

from eegwatch.lslutils import (
    Inlet,
    DataInlet,
    MarkerInlet,
    PULL_INTERVAL,
    PLOT_DURATION,
)

experiment = "test"
subject = "erik"
subject_id = 0
record_duration = 5 * 60  # 5min

# For plotting
UPDATE_INTERVAL = 60  # ms between screen updates


logger = logging.getLogger(__name__)


@click.group(help="Collect EEG data during device usage")
def main():
    logging.basicConfig()
    pass


@main.command()
@click.option(
    "--device", type=click.Choice(["muse2", "openbci"]), help="Which device to use"
)
def connect(device: str):
    # from eegnb.devices.eeg import EEG
    from .devices.eeg import EEG

    # from eegnb import generate_save_fn
    from .util import generate_save_fn

    if device == "muse2":
        # TODO: How can we also get simultaneous HR/HRV tracking?
        board_name = "muse2"
    elif device == "openbci":
        board_name = "openbci"
    else:
        raise ValueError("should be unreachable")

    eeg_device = EEG(device=board_name)

    while True:
        # Create save file name
        save_fn = generate_save_fn(
            board_name, experiment, subject_id=subject_id, session_nb=1
        )

        logger.info(f"Recording to {save_fn}")
        try:
            eeg_device.start(record_duration)
        except IndexError:
            logger.exception("Error while starting recording, trying again in 5s...")
            sleep(5)
        sleep(record_duration)
        logger.info("Done recording")


@main.command()
def plot():
    # print(eeg_device)

    # TODO: Get the live data and do basic stuff to check signal quality, such as:
    #        - Checking signal variance.
    #        - Transforming into the frequency domain.

    streams = pylsl.resolve_stream()
    for s in streams:
        logger.debug(streams)

    inlets: List[Inlet] = []

    # eeg_device.muse_StreamOutlet

    # Create the pyqtgraph window
    pw = pg.plot(title="LSL Plot")
    plt = pw.getPlotItem()
    plt.enableAutoRange(x=False, y=True)

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
            logger.info("Adding data inlet: " + info.name())
            inlets.append(DataInlet(info, plt))
        else:
            logger.info("Don't know what to do with stream " + info.name())

    def scroll():
        """Move the view so the data appears to scroll"""
        # We show data only up to a timepoint shortly before the current time
        # so new data doesn't suddenly appear in the middle of the plot
        fudge_factor = PULL_INTERVAL * 0.002
        plot_time = pylsl.local_clock()
        pw.setXRange(plot_time - PLOT_DURATION + fudge_factor, plot_time - fudge_factor)

    def update():
        # Read data from the inlet. Use a timeout of 0.0 so we don't block GUI interaction.
        mintime = pylsl.local_clock() - PLOT_DURATION
        # call pull_and_plot for each inlet.
        # Special handling of inlet types (markers, continuous data) is done in
        # the different inlet classes.
        for inlet in inlets:
            inlet.pull_and_plot(mintime, plt)

    # create a timer that will move the view every update_interval ms
    update_timer = QtCore.QTimer()
    update_timer.timeout.connect(scroll)
    update_timer.start(UPDATE_INTERVAL)

    # create a timer that will pull and add new data occasionally
    pull_timer = QtCore.QTimer()
    pull_timer.timeout.connect(update)
    pull_timer.start(PULL_INTERVAL)

    import sys

    # Start Qt event loop unless running in interactive mode or using pyside.
    if (sys.flags.interactive != 1) or not hasattr(QtCore, "PYQT_VERSION"):
        QtGui.QApplication.instance().exec_()


if __name__ == "__main__":
    main()
