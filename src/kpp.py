import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.nn.functional import pad
import numpy as np

import torch
from sklearn.cluster import kmeans_plusplus
from numpy.random import choice
import numpy as np
import matplotlib.pyplot as plt
import os


def extract_patches(images, patch_shape, stride, padding=(0, 0), in_order="NCHW", out_order="NCHW"):
    assert images.ndim >= 2 and images.ndim <= 4
    if isinstance(images, np.ndarray):
        from sklearn.feature_extraction.image import _extract_patches

        if images.ndim == 2:  # single gray image
            images = np.expand_dims(images, 0)

        if images.ndim == 3:
            if images.shape[2] == 3:  # single color image
                images = np.expand_dims(images, 0)
            else:  # multiple gray images or single gray image with first index 1
                images = np.expand_dims(images, 3)

        elif in_order == "NCHW":
            images = images.transpose(0, 2, 3, 1)
        # numpy expects order NHWC
        patches = _extract_patches(
            images,
            patch_shape=(1, *patch_shape),
            extraction_step=(1, stride, stride, 1),
            ).reshape(-1, *patch_shape)
        # now patches' shape = NHWC

        if out_order == "NHWC":
            pass
        elif out_order == "NCHW":
            patches = patches.permute(0, 3, 1, 2)
        else:
            raise ValueError(
                'out_order not understood (expected "NHWC" or "NCHW")')

    elif isinstance(images, torch.Tensor):
        if images.ndim == 2:  # single gray image
            images = images.unsqueeze(0)

        if images.ndim == 3:
            if images.shape[2] == 3:  # single color image
                images = images.unsqueeze(0)
            else:  # multiple gray image
                images = images.unsqueeze(3)

        if in_order == "NHWC":
            images = images.permute(0, 3, 1, 2)
        # torch expects order NCHW

        images = pad(images, pad=padding*2)
        # if padding[0] != 0:
        #     breakpoint()

        patches = torch.nn.functional.unfold(
            images, kernel_size=patch_shape[:2], stride=stride
            )
        # at this point patches.shape = N, prod(patch_shape), n_patch_per_img

        # all these operations are done to circumvent pytorch's N,C,H,W ordering
        patches = patches.permute(0, 2, 1)
        n_patches = patches.shape[0] * patches.shape[1]
        patches = patches.reshape(n_patches, patch_shape[2], *patch_shape[:2])
        # now patches' shape = NCHW
        if out_order == "NHWC":
            patches = patches.permute(0, 2, 3, 1)
        elif out_order == "NCHW":
            pass
        else:
            raise ValueError(
                'out_order not understood (expected "NHWC" or "NCHW")')

    return patches


def KMeansPPInitializer(weights, patches, N=1e6):

    patch_shape = patches.shape[1:]
    if len(patches > N):
        patches = patches[choice(len(patches), int(N))]

    patches = patches.reshape(len(patches), -1)

    centers = kmeans_plusplus(patches, len(weights))[0]

    zero_filters = abs(centers).sum(axis=1) == 0.0

    centers[zero_filters] += 1e-6
    centers = centers.reshape(len(centers), *patch_shape)

    with torch.no_grad():
        weights.data = torch.from_numpy(centers).to(weights.device)
