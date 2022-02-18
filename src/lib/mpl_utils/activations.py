import ray
import matplotlib.pyplot as plt
import torch
import os
from pygifsicle import optimize
import numpy as np
from typing import List


def plot_activations(activations: torch.Tensor, filepath: str, file_name: str, suptitle: str = "Activations") -> None:
    r"""
    Plot

    """

    mins = torch.amin(activations, dim=(1, 2), keepdim=True)
    maxs = torch.amax(activations, dim=(1, 2), keepdim=True)
    activations -= mins
    activations /= (maxs-mins+0.0000001)
    mins_ = mins.squeeze().tolist()
    maxs_ = maxs.squeeze().tolist()
    
    # breakpoint()

    n_subplot_sqrt: int = np.ceil(np.sqrt(activations.shape[0])).astype(int)
    # n_subplot_sqrt: int = np.floor(np.sqrt(activations.shape[1])).astype(int)
    fig = plt.figure(figsize=(12, 12))
    for i in range(n_subplot_sqrt):
        for j in range(n_subplot_sqrt):
            if n_subplot_sqrt*i+j == activations.shape[0]:
                break
            plt.subplot(n_subplot_sqrt, n_subplot_sqrt, n_subplot_sqrt*i+j+1)
            plt.imshow(activations[n_subplot_sqrt*i+j].numpy())
            plt.xticks([])
            plt.yticks([])
            plt.xlabel(
                f"M:{maxs_[n_subplot_sqrt*i+j]:.2f} m:{mins_[n_subplot_sqrt*i+j]:.2f}")
        if n_subplot_sqrt*i+j == activations.shape[0]:
            break

    fig.tight_layout()
    fig.suptitle(f"Activations")

    os.makedirs(filepath, exist_ok=True)
    plt.savefig(filepath + file_name + ".pdf")
    plt.close()
