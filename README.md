MSc Thesis
==========

My MSc thesis on "Classifying brain activity using low-cost biosensors and automated time tracking" (working title).

It is very much a work-in-progress. Progress is tracked using the [GitHub Projects board](https://github.com/ErikBjare/thesis/projects/1).

# Usage

Setting it up:

 - Ensure you have Python 3.7+ and `poetry` installed
 - Install dependencies with `poetry install`

Collecting data:

 - Run `eegwatch --help` for usage instructions
 - TODO: Create script to collect categorized data from ActivityWatch

Running classifier:

 - TODO

# Supported devices

 - Muse S 
   - PPG support (experimental)
 - OpenBCI Cyton (WIP)

# Writing

## Thesis

The latest version of the thesis can be downloaded at https://erik.bjareholt.com/thesis/thesis.pdf

## Goal Document

The latest version of the goal document can be downloaded at https://erik.bjareholt.com/thesis/goaldocument.pdf

# Resources

 - MNE: https://mne.tools/
 - PsychoPy: https://www.psychopy.org/
   - Invaluable for running experiments.
 - https://github.com/NeuroTechX/eeg-notebooks
   - Have examples with sklearn + riemannian geometry.
 - Overview of public EEG datasets: https://github.com/meagmohit/EEG-Datasets
 - [braindecode](https://github.com/braindecode/braindecode): A deep learning toolbox to decode raw time-domain EEG. 
