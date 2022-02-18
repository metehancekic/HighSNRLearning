"""
Description: Training and testing functions for neural models

functions:
    train: Performs a single training epoch (if attack_args is present adversarial training)
    test: Evaluates model by computing accuracy (if attack_args is present adversarial testing)
"""

from tqdm import tqdm
import numpy as np
from functools import partial
from time import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from .regularizer import matching_loss, l1_loss
from .augmentation import get_noisy_images
from .surgery import extract_patches


def single_epoch(cfg, 
                 model, 
                 train_loader, 
                 optimizer, 
                 scheduler=None, 
                 compute_activation_disparity = True, 
                 logging_func=print, 
                 verbose: bool = True, 
                 epoch: int = 0):
    r"""
    Single epoch
    """
    start_time = time()
    model.train()

    device = model.parameters().__next__().device

    train_loss = 0
    train_correct = 0
    if compute_activation_disparity or "matching" in cfg.train.regularizer.active:
        activation_disparity = torch.zeros(len(model.layers_of_interest.values())).to(device)
    for data, target in train_loader:

        data, target = data.to(device), target.to(device)

        loss = torch.Tensor([0.0]).to(device)
        optimizer.zero_grad()
        if compute_activation_disparity or "matching" in cfg.train.regularizer.active:
            _ = model(data)
            top_K_activations_per_layer = torch.zeros(len(model.layers_of_interest)).to(device)
            bottom_activations_per_layer = torch.zeros(len(model.layers_of_interest)).to(device)
            matching_loss_per_layer = torch.zeros(len(model.layers_of_interest)).to(device)
            for idx, (layer, layer_input, layer_output) in enumerate(zip(model.layers_of_interest.values(), model.layer_inputs.values(), model.layer_outputs.values())):
                loss_args = dict(activations=layer_output,
                                    ratio=cfg.train.regularizer.matching.ratio,
                                    saliency_lambda=cfg.train.regularizer.matching.lamda,
                                    dim=cfg.train.regularizer.matching.dim)
                if isinstance(layer, nn.Conv2d):
                    patch_extractor = partial(extract_patches, patch_shape=(*layer.kernel_size, layer.in_channels),
                                                stride=layer.stride, padding=layer.padding*2, in_order="NCHW", out_order="NCHW")
                    patches = patch_extractor(layer_input)
                    loss_args["patches"] = patches
                    if type(layer) == nn.Conv2d:
                        loss_args["weights"] = layer.weight

                top_K_activations_per_layer[idx], bottom_activations_per_layer[idx], matching_loss_per_layer[idx] = matching_loss(**loss_args)
            with torch.no_grad():
                activation_disparity += top_K_activations_per_layer-bottom_activations_per_layer
            if "matching" in cfg.train.regularizer.active:
                # if cfg.train.reg.matching.style == "scalar":
                #     loss -= cfg.train.reg.matching.scale * torch.mean(top_K_layer - cfg.train.reg.matching.lamda * bottom_layer)
                # elif cfg.train.reg.matching.style == "vector":
                # scalar_vector = torch.Tensor([3.0, 2.0, 1.0, 0.5, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).to(device)/10.0
                # scalar_vector = torch.Tensor([0.1, 0.02, 0.005, 0.000001, 0.000001, 0.00000001, 0.0, 0.0, 0.0, 0.0, 0.0, -0.01, -0.01]).to(device)
                # scalar_vector = torch.Tensor([0.06, 0.015, 0.005, 0.000001, 0.000001, 0.00000001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).to(device) # 123
                # scalar_vector = torch.Tensor([0.06, 0.015, 0.005, 0.000001, 0.000001, 0.00000001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).to(device) # 246
                # scalar_vector = torch.Tensor([0.07, 0.016, 0.009, 0.000001, 0.000001, 0.00000001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).to(device) # 345
                # scalar_vector = torch.Tensor([0.07, 0.016, 0.006, 0.000001, 0.000001, 0.00000001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).to(device) # 456
                # scalar_vector = torch.Tensor([0.07, 0.016, 0.008, 0.0001, 0.000001, 0.00000001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).to(device) # 567
                # scalar_vector = torch.Tensor([0.07, 0.017, 0.009, 0.0001, 0.000001, 0.00000001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).to(device) # 678
                # scalar_vector = torch.Tensor([0.07, 0.018, 0.01, 0.0001, 0.000001, 0.00000001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).to(device) # 789
                # scalar_vector = torch.Tensor([0.066, 0.0177, 0.009, 0.0001, 0.000001, 0.00000001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).to(device) # 891
                # scalar_vector = torch.Tensor([0.07, 0.017, 0.009, 0.0001, 0.000001, 0.00000001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).to(device) #
                scalar_vector = torch.Tensor(cfg.train.regularizer.matching.alpha).to(device) # 999
                # scalar_vector = -torch.ones(13).to(device)
        #         tensor([0.2552, 0.2408, 0.0697, 0.0545, 0.0419, 0.0325, 0.0834, 0.0729, 0.0542,                                                                                                              │tensor([0.6878, 0.0250, 0.4025, 0.0443, 0.0436, 0.0387, 0.0976, 0.0581, 0.0498,
        # 0.1275, 0.1720, 0.1572, 0.3229], device='cuda:0')
                # breakpoint()
                # loss -= torch.mean(scalar_vector*(top_K_layer - cfg.train.reg.matching.lamda * bottom_layer))
                loss -= torch.mean(scalar_vector * matching_loss_per_layer)

        if cfg.train.type == "noisy":
            data = get_noisy_images(data, cfg.train.noise.std)

        if cfg.train.type == "adversarial":
            from deepillusion.torchattacks import FGSM, RFGSM, PGD
            perturbs = locals()[cfg.train.adversarial.attack](net=model, x=data, y_true=target, data_params={
                "x_min": cfg.dataset.min, "x_max": cfg.dataset.max}, attack_params=cfg.train.adversarial, verbose=False)
            data += perturbs

        output = model(data)
        cross_ent = nn.CrossEntropyLoss()
        xent_loss = cross_ent(output, target)

        l1_weight_loss = 0
        if "l1_weight" in cfg.train.regularizer.active:
            for _, layer in model.named_modules():
                if isinstance(layer, torch.nn.Conv2d):
                    l1_weight_loss += l1_loss(features={"conv": layer.weight}, dim=(1, 2, 3))

        loss += xent_loss + cfg.train.regularizer.l1_weight.scale * l1_weight_loss

        loss.backward()
        optimizer.step()
        if scheduler and cfg.nn.scheduler == "cyc":
            scheduler.step()

        train_loss += xent_loss.item() * data.size(0)
        pred_adv = output.argmax(dim=1, keepdim=False)
        train_correct += pred_adv.eq(target.view_as(pred_adv)).sum().item()

    if scheduler and not cfg.nn.scheduler == "cyc":
        scheduler.step()

    train_size = len(train_loader.dataset)
    train_loss = train_loss/train_size
    train_acc = train_correct/train_size
    activation_disparity = activation_disparity/len(train_loader)


    if verbose:
        logging_func(f"Epoch: \t {epoch} \t Time (s): {(time()-start_time):.0f}")
        logging_func(f"Train Xent loss: \t {train_loss:.2f} \t Train acc: {100*train_acc:.2f} %")
        logging_func(f"L1 Weight Loss: \t {l1_weight_loss:.4f}")
        if compute_activation_disparity or "matching" in cfg.train.regularizer.active:
            logging_func(f"Mean Activation disparity: \t {torch.mean(activation_disparity):.4f}")
            logging_func(f"Activation Disparity per Layer: {activation_disparity}")
        logging_func("-"*100)



def standard_test(model, test_loader, verbose=True, progress_bar=False):
    """
    Description: Evaluate model with test dataset,
        if adversarial args are present then adversarially perturbed test set.
    Input :
        model : Neural Network               (torch.nn.Module)
        test_loader : Data loader            (torch.utils.data.DataLoader)
        verbose: Verbosity                   (Bool)
        progress_bar: Progress bar           (Bool)
    Output:
        train_loss : Train loss              (float)
        train_accuracy : Train accuracy      (float)
    """

    device = model.parameters().__next__().device

    model.eval()

    test_loss = 0
    test_correct = 0
    if progress_bar:
        iter_test_loader = tqdm(
            iterable=test_loader,
            unit="batch",
            leave=False)
    else:
        iter_test_loader = test_loader

    for data, target in iter_test_loader:

        data, target = data.to(device), target.to(device)

        output = model(data)

        cross_ent = nn.CrossEntropyLoss()
        test_loss += cross_ent(output, target).item() * data.size(0)

        pred = output.argmax(dim=1, keepdim=False)
        test_correct += pred.eq(target.view_as(pred)).sum().item()

    test_size = len(test_loader.dataset)
    if verbose:
        print(f"Test loss: {test_loss/test_size:.4f}, Test acc: {100*test_correct/test_size:.2f}")

    return test_loss/test_size, test_correct/test_size

def matching_test(cfg, model, test_loader, verbose=False, progress_bar=False):
    """
    Description: Evaluate model with test dataset,
        if adversarial args are present then adversarially perturbed test set.
    Input :
        model : Neural Network               (torch.nn.Module)
        test_loader : Data loader            (torch.utils.data.DataLoader)
        verbose: Verbosity                   (Bool)
        progress_bar: Progress bar           (Bool)
    Output:
        train_loss : Train loss              (float)
        train_accuracy : Train accuracy      (float)
    """

    device = model.parameters().__next__().device

    model.eval()

    test_loss = 0
    test_correct = 0
    if progress_bar:
        iter_test_loader = tqdm(
            iterable=test_loader,
            unit="batch",
            leave=False)
    else:
        iter_test_loader = test_loader

    top_K_minus_bottom = 0
    counter = 0
    for data, target in iter_test_loader:
        counter += 1
        data, target = data.to(device), target.to(device)

        output = model(data)
        top_K_layer = torch.zeros(len(model.layers_of_interest))
        bottom_layer = torch.zeros(len(model.layers_of_interest))
        for idx, (layer, layer_input, layer_output) in enumerate(zip(model.layers_of_interest.values(), model.layer_inputs.values(), model.layer_outputs.values())):
            loss_args = dict(activations=layer_output,
                             ratio=cfg.test.matching.ratio,
                             saliency_lambda=1.0,
                             dim=cfg.test.matching.dim)
            if isinstance(layer, nn.Conv2d):
                patch_extractor = partial(extract_patches, patch_shape=(*layer.kernel_size, layer.in_channels),
                                            stride=layer.stride, padding=layer.padding*2, in_order="NCHW", out_order="NCHW")

                patches = patch_extractor(layer_input)

                loss_args["patches"] = patches
                if type(layer) == nn.Conv2d:
                    loss_args["weights"] = layer.weight

            top_K_layer[idx], bottom_layer[idx], _ = matching_loss(**loss_args)
        with torch.no_grad():
            top_K_minus_bottom += top_K_layer-bottom_layer
        
        cross_ent = nn.CrossEntropyLoss()
        test_loss += cross_ent(output, target).item() * data.size(0)

        pred = output.argmax(dim=1, keepdim=False)
        test_correct += pred.eq(target.view_as(pred)).sum().item()

    top_K_minus_bottom /= counter

    test_size = len(test_loader.dataset)
    if verbose:
        print(f"Test loss: {test_loss/test_size:.4f}, Test acc: {100*test_correct/test_size:.2f}, Test matching: {top_K_minus_bottom:.4f}")

    return test_loss/test_size, test_correct/test_size, top_K_minus_bottom
