import os
from pathlib import Path

from . import load

src_dir = Path(os.path.dirname(os.path.realpath(__file__)))

data_dir = src_dir.parent.parent / "data"
