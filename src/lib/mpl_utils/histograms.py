import ray
import matplotlib.pyplot as plt
import torch
import os
from pygifsicle import optimize
import numpy as np
from typing import List


def plot_histograms(activations: torch.Tensor, filepath: str, file_name: str, suptitle: str = "Activations") -> None:
    r"""
    Plot

    """

    n_subplot_sqrt: int = np.ceil(np.sqrt(activations.shape[0])).astype(int)
    # n_subplot_sqrt: int = np.floor(np.sqrt(activations.shape[1])).astype(int)
    fig = plt.figure(figsize=(12, 12))
    for i in range(n_subplot_sqrt):
        for j in range(n_subplot_sqrt):
            if n_subplot_sqrt*i+j == activations.shape[0]:
                break
            filter_idx = n_subplot_sqrt*i+j
            plt.subplot(n_subplot_sqrt, n_subplot_sqrt, n_subplot_sqrt*i+j+1)
            abs_max = np.abs(activations.numpy()[filter_idx]).max()
            xlims = (0, abs_max+0.0001)

            bin_edges = np.linspace(*xlims, 50)

            hist, _ = np.histogram(activations.numpy()[filter_idx], bin_edges, density=True)

            color, edgecolor = ("orange", "darkorange")

            plt.bar(
                bin_edges[:-1] + np.diff(bin_edges) / 2,
                hist,
                width=(bin_edges[1] - bin_edges[0]),
                alpha=0.5,
                edgecolor="none",
                color=color,
                )
            plt.step(
                np.array([*bin_edges, bin_edges[-1] + (bin_edges[1] - bin_edges[0])]),
                np.array([0, *hist, 0]),
                label=f"Filter {i}",
                where="pre",
                color=edgecolor,
                )
            ax = plt.gca()
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.get_yaxis().set_visible(False)
            plt.grid()
        if n_subplot_sqrt*i+j == activations.shape[0]:
            break

    fig.tight_layout()
    fig.suptitle(f"Activations")

    os.makedirs(filepath, exist_ok=True)
    plt.savefig(filepath + file_name + ".pdf")
