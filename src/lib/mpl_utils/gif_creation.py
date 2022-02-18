import ray
import matplotlib.pyplot as plt
import torch
from imageio import mimsave
import os
from pygifsicle import optimize
import numpy as np
from typing import List


# @ray.remote(num_returns=1)
def process_weights(weight_list: List[torch.Tensor], k: int):
    weights = weight_list[k]
    mins = torch.amin(weights, dim=(1, 2, 3), keepdim=True)
    maxs = torch.amax(weights, dim=(1, 2, 3), keepdim=True)
    weights -= mins
    weights /= (maxs-mins)
    mins_ = mins.squeeze().tolist()
    maxs_ = maxs.squeeze().tolist()

    n_subplot_sqrt: int = np.ceil(np.sqrt(len(weights))).astype(int)
    # n_subplot_sqrt: int = np.floor(np.sqrt(len(weights))).astype(int)
    fig = plt.figure(figsize=(12, 12))
    for i in range(n_subplot_sqrt):
        for j in range(n_subplot_sqrt):
            if n_subplot_sqrt*i+j == len(weights):
                break
            plt.subplot(n_subplot_sqrt, n_subplot_sqrt, n_subplot_sqrt*i+j+1)
            plt.imshow(
                weights[n_subplot_sqrt*i+j].permute(1, 2, 0).numpy())
            plt.xticks([])
            plt.yticks([])
            plt.xlabel(
                f"M:{maxs_[n_subplot_sqrt*i+j]:.2f} m:{mins_[n_subplot_sqrt*i+j]:.2f}")
        if n_subplot_sqrt*i+j == len(weights):
            break

    # fig.tight_layout()
    fig.suptitle(f"Epoch {k}")
    fig.canvas.draw()
    image_from_plot: np.ndarray = np.frombuffer(
        fig.canvas.tostring_rgb(), dtype=np.uint8)
    ncols, nrows = fig.canvas.get_width_height()[::-1]
    try:
        image_from_plot = image_from_plot.reshape(nrows, ncols, 3)
    except:
        image_from_plot = image_from_plot.reshape(2*nrows, 2*ncols, 3)

    plt.close()

    return image_from_plot


def save_gif(weight_list: List[torch.Tensor], filepath: str, file_name: str, num_cpus: int = 10):

    # try:
    #     ray.init(num_cpus=num_cpus, log_to_driver=False)
    # except:
    #     pass
    # weight_list_id = ray.put(weight_list)

    # result_ids = [process_weights.remote(
    #     weight_list_id, i) for i in range(len(weight_list))]

    # gif_images = ray.get(result_ids)

    gif_images=[process_weights(weight_list, i) for i in range(len(weight_list))]

    if not os.path.exists(filepath):
        os.makedirs(filepath)

    mimsave(filepath + file_name + ".gif",
            gif_images, duration=0.1)

    optimize(filepath + file_name + ".gif")
