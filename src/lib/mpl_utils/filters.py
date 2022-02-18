import ray
import matplotlib.pyplot as plt
import torch
import os
from pygifsicle import optimize
import numpy as np
from typing import List


def plot_filters(filters: torch.Tensor, filepath: str, file_name: str, suptitle: str = "Filters") -> None:
    r"""
    Plot

    """

    mins = torch.amin(filters, dim=(1, 2), keepdim=True)
    maxs = torch.amax(filters, dim=(1, 2), keepdim=True)
    filters -= mins
    filters /= (maxs-mins)
    mins_ = mins.squeeze().tolist()
    maxs_ = maxs.squeeze().tolist()

    n_subplot_sqrt: int = np.ceil(np.sqrt(filters.shape[0])).astype(int)
    fig = plt.figure(figsize=(12, 12))
    for i in range(n_subplot_sqrt):
        for j in range(n_subplot_sqrt):
            if n_subplot_sqrt*i+j == filters.shape[0]:
                break
            plt.subplot(n_subplot_sqrt, n_subplot_sqrt, n_subplot_sqrt*i+j+1)
            plt.imshow(filters[n_subplot_sqrt*i+j].numpy())
            plt.xticks([])
            plt.yticks([])
            plt.xlabel(
                f"M:{maxs_[n_subplot_sqrt*i+j]:.2f} m:{mins_[n_subplot_sqrt*i+j]:.2f}")
        if n_subplot_sqrt*i+j == filters.shape[0]:
            break

    fig.tight_layout()
    fig.suptitle(suptitle)

    os.makedirs(filepath, exist_ok=True)
    plt.savefig(filepath + file_name + ".pdf")
