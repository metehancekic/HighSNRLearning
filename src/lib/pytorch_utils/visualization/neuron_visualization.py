from typing import Dict, Iterable, Union, Tuple

import matplotlib.pyplot as plt
import os
from tqdm import tqdm

import torch
from torch.optim import Adam
from torch import nn
from torch.autograd import Variable


from ..surgery import LayerOutputExtractor_wrapper


def neuron_maximizer(model: torch.nn.Module, data_loader, layer_name: str, filter_num: int = None) -> torch.Tensor:
    """
    inputs:
        model: torch.nn.module
        data_loader:
        layer_name:
        filter_num:
    outputs:
        maximizing_image:
    """

    device = model.parameters().__next__().device

    maximum_activation = 0
    maximizing_image = torch.zeros((1, 1, 28, 28))

    for data, target in data_loader:
        data, target = data.to(device), target.to(device)

        _ = model(data)

        layer_output = model.layer_outputs[layer_name]

        layer_shape = layer_output.shape

        if filter_num:
            current_max, current_idx = torch.max(
                layer_output[:, filter_num, layer_shape[2]//2, layer_shape[3]//2], dim=0)
        else:
            current_max, current_idx = torch.max(
                layer_output[:, layer_shape[1]//2, layer_shape[2]//2, layer_shape[3]//2], dim=0)
            filter_num = layer_shape[1]//2

        if current_max > maximum_activation:
            maximizing_image = data[current_idx, ...].unsqueeze(0).detach().cpu()

    return maximizing_image


def neuron_visualizer(model: torch.nn.Module, layer_name: str, data_loader, starting_image: str = "max", dataset: str = "mnist",  verbose: bool = False):
    """
    Description: Feature Visualizer for a given neuron
    Input :
        net : Neural Network                                        (torch.nn.Module)
    Output:
        perturbation : Single step perturbation (Clamped with input limits)

    Explanation:
        e = epsilon * sign(grad_{x}(net(x)))
    """

    device = model.parameters().__next__().device
    model = LayerOutputExtractor_wrapper(model, layer_names=[layer_name])

    if dataset == "mnist":
        image = (torch.rand(1, 1, 28, 28)/10)
    elif dataset == "cifar":
        image = (torch.rand(1, 3, 32, 32)/10)
    else:
        raise NotImplementedError

    _ = model(image.to(device))
    layer_output = model.layer_outputs[layer_name]
    layer_shape = layer_output.shape

    if starting_image == "max":
        images = [None]*layer_shape[1]
        for i in tqdm(range(layer_shape[1])):
            images[i] = neuron_maximizer(model=model, data_loader=data_loader,
                                         layer_name=layer_name, filter_num=i)

    visualizations = [None]*layer_shape[1]
    for filter in tqdm(range(layer_shape[1])):
        image = images[filter].detach().clone()
        
        for i in range(1, 31):
            image = Variable(image, requires_grad=True)
            optimizer = Adam([image], lr=0.01, weight_decay=1e-6)
            optimizer.zero_grad()
            _ = model(image.to(device))

            layer_output = model.layer_outputs[layer_name]

            layer_shape = layer_output.shape

            loss = -layer_output[0, filter, layer_shape[2]//2, layer_shape[3]//2]

            loss.backward()
            optimizer.step()

            image = torch.clamp(image, min=0.0, max=1.0)
            # breakpoint()

        visualizations[filter] = image[0].permute(1, 2, 0).detach().cpu().numpy()
        # visualizations[filter] = normalize_image(image)[0].permute(1, 2, 0).detach().cpu().numpy()

    images = [image[0].permute(1, 2, 0).detach().cpu().numpy() for image in images]
    return images, visualizations


def filter_visualizer(model: torch.nn.Module, layer_name: str, filter_num: int = None, save_loc: str = "", starting_image: torch.Tensor = None, dataset: str = "mnist",  verbose: bool = False):
    """
    Description: Feature Visualizer for a given neuron
    Input :
        net : Neural Network                                        (torch.nn.Module)
    Output:
        perturbation : Single step perturbation (Clamped with input limits)

    Explanation:
        e = epsilon * sign(grad_{x}(net(x)))
    """
    device = model.parameters().__next__().device
    model = LayerOutputExtractor_wrapper(model, layer_names=[layer_name])

    if starting_image != None:
        if dataset == "mnist":
            starting_image = (torch.rand(1, 1, 28, 28)/10)
        elif dataset == "cifar":
            starting_image = (torch.rand(1, 3, 32, 32)/10)
        else:
            raise NotImplementedError

    starting_image = starting_image.to(device)

    image = Variable(starting_image, requires_grad=True)
    optimizer = Adam([image], lr=0.01, weight_decay=1e-6)

    os.makedirs(save_loc, exist_ok=True)
    im_path = save_loc + "starting_image" + '.jpg'
    plt.figure()
    plt.imshow(image[0].permute(1, 2, 0).detach().cpu().numpy())
    plt.savefig(im_path)
    plt.close()

    for i in range(1, 31):
        optimizer.zero_grad()
        _ = model(image)

        layer_output = model.layer_outputs[layer_name]

        layer_shape = layer_output.shape

        if filter_num:
            loss = -torch.mean(layer_output[0, filter_num])
        else:
            loss = -torch.mean(layer_output[0, layer_shape[1]//2])
            filter_num = layer_shape[1]//2

        loss.backward()
        optimizer.step()

        if i % 5 == 0:
            im_path = save_loc + model.name + "_" + str(layer_name) + \
                '_f_' + str(filter_num) + '_iter_' + str(i) + '.jpg'
            plt.figure()
            plt.imshow(normalize_image(image)[0].permute(1, 2, 0).detach().cpu().numpy())
            plt.savefig(im_path)
            plt.close()


def normalize_image(image: torch.Tensor):
    return (image-torch.min(image))/(torch.max(image)-torch.min(image)+1e-9)
