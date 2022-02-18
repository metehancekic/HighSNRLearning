
from typing import Union, Tuple, Sequence

import torch
import numpy as np
from torch.nn.functional import pad


def extract_patches(
    images: Union[np.ndarray, torch.Tensor],
    patch_shape: Tuple[int, int, int],
    stride: int,
    padding: Sequence[int] = (0, 0),
    in_order="NHWC",
    out_order="NHWC"
):
    assert images.ndim >= 2 and images.ndim <= 4

    patches = extract_patches_torch(
        images, patch_shape, stride, padding=padding, in_order=in_order, out_order=out_order)

    return patches


def extract_patches_torch(
    images: torch.Tensor,
    patch_shape: Tuple[int, int, int],
    stride: int,
    padding: Sequence[int] = (0, 0),
    in_order="NHWC",
    out_order="NHWC"
):

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

    images = pad(images, pad=padding)
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
