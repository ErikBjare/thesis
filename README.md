MSc Thesis
==========

[![GitHub Actions badge](https://github.com/ErikBjare/thesis/workflows/Test/badge.svg)](https://github.com/ErikBjare/thesis/actions)
[![Code coverage](https://codecov.io/gh/ErikBjare/thesis/branch/master/graph/badge.svg)](https://codecov.io/gh/ErikBjare/thesis)
[![Typechecking: Mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

My MSc thesis on "Classifying brain activity using low-cost biosensors and automated time tracking" (working title).

It is very much a work-in-progress. Progress is tracked using the [GitHub Projects board](https://github.com/ErikBjare/thesis/projects/1).

# Usage

Setting it up:

 - Ensure you have Python 3.7+ and `poetry` installed
 - Install dependencies with `poetry install`

Collecting data:

 - Run `eegwatch --help` for instructions on how to collect EEG data
 - Run [ActivityWatch](https://activitywatch.net) to collect device activity data
 - Run the codeprose task in [eeg-notebooks][eegnb] to collect data for the code vs prose task

Running classifier:

 - Run `./scripts/query_aw.py` to collect labels from the running ActivityWatch instance
   - You probably want to adjust the categorization rules embedded in the file
 - (TODO) Run `eegclassify --help` for instructions on how to train and run the classifier

# Supported devices

 - Muse S 
   - PPG support (experimental)
 - Neurosity Notion 1 & 2
   - Thanks to [@andrewjaykeller](https://github.com/andrewjaykeller) at [@neurosity](https://github.com/neurosity) for sending me a refurbished DK1 to test with!
 - OpenBCI Cyton (WIP)
 - In theory: any device supported by Brainflow or muse-lsl

# Notebooks

Code notebooks are built in CI and available at:

 - [Main][nbmain] - primary notebook for the thesis, where we train a classifier for the code vs prose comprehension task.
 - [Signal][nbsignal] - for signal filtering and quality checking.
 - [Activity][nbactivity] - for classification of device activities.

[nbmain]:       https://erik.bjareholt.com/thesis/Main.html
[nbsignal]:     https://erik.bjareholt.com/thesis/Signal.html
[nbactivity]:   https://erik.bjareholt.com/thesis/Activity.html

# Writing

## Thesis

The latest version of the thesis can be downloaded at [erik.bjareholt.com/thesis/thesis.pdf][thesis]

## Goal Document

The latest version of the goal document can be downloaded at [erik.bjareholt.com/thesis/goaldocument.pdf][goaldoc]

# Acknowledgements

See the Acknowledgements section in the [thesis][thesis].

[thesis]: https://erik.bjareholt.com/thesis/thesis.pdf
[goaldoc]: https://erik.bjareholt.com/thesis/goaldocument.pdf

# Resources

 - MNE: https://mne.tools/
 - PsychoPy: https://www.psychopy.org/
   - Invaluable for running experiments.
 - [eeg-notebooks][eegnb]
   - Have examples with sklearn + riemannian geometry.
 - Overview of public EEG datasets: https://github.com/meagmohit/EEG-Datasets
 - [braindecode](https://github.com/braindecode/braindecode): A deep learning toolbox to decode raw time-domain EEG. 

[eegnb]: https://github.com/NeuroTechX/eeg-notebooks
