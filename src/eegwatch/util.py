import os
import time

DATA_DIR = os.path.join(os.path.expanduser("~/"), ".eegnb", "data")


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
    recording_dir = os.path.join(
        data_dir, experiment, "local", board_name, subject_str, session_str
    )

    # check if directory exists, if not, make the directory
    if not os.path.exists(recording_dir):
        os.makedirs(recording_dir)

    # generate filename based on recording date-and-timestamp and then append to recording_dir
    save_fp = os.path.join(
        recording_dir,
        ("recording_%s" % time.strftime("%Y-%m-%d-%H.%M.%S", time.gmtime()) + ".csv"),
    )

    return save_fp
