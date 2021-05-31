import logging
from typing import List, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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


class TimelineFigure:
    def __init__(self, title=None, **kwargs):
        self.fig = plt.figure(**kwargs)
        self.ax = plt.gca()
        if title:
            self.ax.set_title(title)
        self.bars = []

    def plot(self):
        for bar_idx, bar in enumerate(self.bars):
            for event in bar["events"]:
                start, end, color = event
                plt.barh(-bar_idx, end - start, left=start, color=color)

        tick_idxs = list(range(0, -len(self.bars), -1))
        self.ax.set_yticks(tick_idxs)
        self.ax.set_yticklabels([bar["title"] for bar in self.bars])

        plt.show()

    def add_bar(self, events: Tuple[datetime, datetime, str], title: str):
        self.bars.append({"events": events, "title": title})
