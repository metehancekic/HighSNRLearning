import matplotlib.pyplot as plt
import torch
import os
import numpy as np
from typing import List


def plot_tensors(tensor_list: List[torch.Tensor], filepath: str, file_name: str, title: str):
    mins = [np.min(tensor) for tensor in tensor_list]
    maxs = [np.max(tensor) for tensor in tensor_list]

    tensor_list = [(tensor_list[i]-mins[i])/(maxs[i]-mins[i]+1e-9) for i in range(len(tensor_list))]

    n_subplot_sqrt: int = np.floor(np.sqrt(len(tensor_list))).astype(int)
    fig = plt.figure(figsize=(12, 12))
    for i in range(n_subplot_sqrt):
        for j in range(n_subplot_sqrt):
            plt.subplot(n_subplot_sqrt, n_subplot_sqrt, n_subplot_sqrt*i+j+1)
            plt.imshow(tensor_list[n_subplot_sqrt*i+j])
            plt.xticks([])
            plt.yticks([])
            plt.xlabel(
                f"M:{maxs[n_subplot_sqrt*i+j]:.2f} m:{mins[n_subplot_sqrt*i+j]:.2f}")

    fig.suptitle(title)
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    plt.savefig(filepath + file_name + ".pdf")
    plt.close()


def plot_images(images: np.array, filepath: str, file_name: str):

    n_subplot_sqrt: int = np.floor(np.sqrt(len(images))).astype(int)
    fig = plt.figure(figsize=(12, 12))
    for i in range(n_subplot_sqrt):
        for j in range(n_subplot_sqrt):
            plt.subplot(n_subplot_sqrt, n_subplot_sqrt, n_subplot_sqrt*i+j+1)
            plt.imshow(images[n_subplot_sqrt*i+j])
            plt.xticks([])
            plt.yticks([])

    if not os.path.exists(filepath):
        os.makedirs(filepath)
    plt.savefig(filepath + file_name + ".pdf")
    plt.close()