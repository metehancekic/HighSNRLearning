import matplotlib.pyplot as plt
import torch
import os
import numpy as np


def plot_bar(x, activations, filepath: str, file_name: str, title: str="", color: str="orange"):
    fig = plt.figure(figsize=(12, 12))
    plt.bar(
        x,
        activations,
        width=1,
        alpha=0.5,
        edgecolor="none",
        color=color,
        )
    ax = plt.gca()
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    # ax.get_yaxis().set_visible(False)
    plt.grid()
    plt.title(title)

    plt.tight_layout()

    os.makedirs(filepath, exist_ok=True)
    plt.savefig(filepath + file_name + ".pdf")
