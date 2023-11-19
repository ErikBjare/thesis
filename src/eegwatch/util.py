import os
import sys
import time
from pathlib import Path

DATA_DIR = Path.home() / ".eegnb" / "data"


def generate_save_fn(
    board_name: str,
    experiment: str,
    subject_id: int,
    session_nb: int,
    data_dir=DATA_DIR,
):
    """
    Generates a file name with the proper trial number for the current subject/experiment combo

    Based on: https://github.com/NeuroTechX/eeg-notebooks/blob/926a63c7f4285ebda72f80a907a552ccf030961e/eegnb/__init__.py
    """
    # TODO: Merge back upstream once https://github.com/NeuroTechX/eeg-notebooks/issues/19 is resolved

    # convert subject ID to 4-digit number
    subject_str = "subject%04.f" % subject_id
    session_str = "session%03.f" % session_nb

    # folder structure is /DATA_DIR/experiment/site/subject/session/*.csv
    recording_dir = (
        data_dir / experiment / "local" / board_name / subject_str / session_str
    )

    # create the directory if it doesn't exist
    recording_dir.mkdir(parents=True, exist_ok=True)

    return recording_dir / (
        f'recording_{time.strftime("%Y-%m-%d-%H.%M.%S", time.gmtime())}'
        + ".csv"
    )


def print_statusline(msg: str):
    """From: https://stackoverflow.com/a/43952192/965332"""
    last_msg_length = (
        len(print_statusline.last_msg) if hasattr(print_statusline, "last_msg") else 0  # type: ignore
    )
    print(" " * last_msg_length, end="\r")
    print(msg, end="\r")
    sys.stdout.flush()  # Some say they needed this, I didn't.
    print_statusline.last_msg = msg  # type: ignore
