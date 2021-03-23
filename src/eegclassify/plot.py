import logging
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def classdistribution(df: pd.DataFrame):
    plt.figure(figsize=(12, 5))
    sns.countplot(x=df["class"], color="mediumseagreen")
    plt.title("Class distribution", fontsize=16)
    plt.ylabel("Class Counts", fontsize=16)
    plt.xlabel("Class Label", fontsize=16)
    plt.xticks(rotation="vertical")
    plt.show()


def pca(X: np.ndarray, y: np.ndarray):
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
