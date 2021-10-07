import logging
from typing import List, Optional, Tuple, Iterable, TypeVar, Callable, Union
from datetime import datetime
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from eegclassify.util import take_until_next

logger = logging.getLogger(__name__)


def classdistribution(df: pd.DataFrame) -> None:
    plt.figure(figsize=(12, 5))
    sns.countplot(x=df["class"], color="mediumseagreen")
    plt.title("Class distribution", fontsize=16)
    plt.ylabel("Class Counts", fontsize=16)
    plt.xlabel("Class Label", fontsize=16)
    plt.xticks(rotation="vertical")
    plt.show()


def pca(X: np.ndarray, y: np.ndarray) -> None:
    scaler = StandardScaler()
    scaled_df = scaler.fit_transform(X)
    pca = PCA(n_components=20)
    pca_vectors = pca.fit_transform(scaled_df)
    for index, var in enumerate(pca.explained_variance_ratio_):
        logger.info(f"Explained Variance ratio by PC {index + 1}: {var}")

    plt.figure(figsize=(25, 8))
    sns.scatterplot(x=pca_vectors[:, 0], y=pca_vectors[:, 1], hue=y)
    plt.title("Principal Components vs Class distribution", fontsize=16)
    plt.ylabel("Principal Component 2", fontsize=16)
    plt.xlabel("Principal Component 1", fontsize=16)
    plt.xticks(rotation="vertical")
    plt.show()


T = TypeVar("T")
Color = Union[str, Tuple[float, float, float]]
Index = int  # Union[int, datetime]
Event = Tuple[Index, Index, Color, str]


@dataclass
class Bar:
    title: str
    events: List[Event]
    show_label: bool


class TimelineFigure:
    def __init__(self, title=None, **kwargs):
        self.fig = plt.figure(**kwargs)
        self.ax = plt.gca()
        if title:
            self.ax.set_title(title)
        self.bars: List[Bar] = []

    def plot(self):
        max_end = 0
        for bar_idx, bar in enumerate(self.bars):
            for event in bar.events:
                start, end, color, label = event
                length = end - start
                plt.barh(-bar_idx, length, left=start, color=color)
                max_end = max(max_end, end)
                plt.text(
                    start + length / 2,
                    -bar_idx,
                    label,
                    horizontalalignment="center",
                    verticalalignment="center",
                )

        tick_idxs = list(range(0, -len(self.bars), -1))
        self.ax.set_yticks(tick_idxs)
        self.ax.set_yticklabels([bar.title for bar in self.bars])
        self.ax.set_xlim(0, max_end)

        plt.show()

    def add_bar(self, events: List[Event], title: str, show_label: bool = False):
        self.bars.append(Bar(title, events, show_label))

    def add_chunked(
        self,
        ls: Iterable[T],
        cmap: Callable[[T], Color],
        title: str,
        show_label: bool = False,
    ):
        """Optimized version of add_bar that takes care of identical subsequent values"""
        bars = [
            (i_start, i_end + 1, cmap(v), str(v) if show_label else "")
            for i_start, i_end, v in take_until_next(ls)
        ]
        self.add_bar(bars, title, show_label)
