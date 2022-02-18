"""
"""

import numpy as np
from omegaconf import DictConfig, OmegaConf
import hydra
import time
from os.path import join
from matplotlib import rc
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch


# PYTORCH UTILS
from pytorch_utils.surgery import LayerOutputExtractor_wrapper, SpecificLayerTypeOutputExtractor_wrapper
from pytorch_utils.analysis import count_parameter

from mpl_utils.activations import plot_activations
from mpl_utils import plot_histograms

from .models.custom_models import Custom_LeNet
from .models import LeNet

# Initializers
from .init import *

from .utils.namers import classifier_ckpt_namer, classifier_params_string


@hydra.main(config_path="/home/metehan/icip/src/configs", config_name="cifar")
def main(cfg: DictConfig) -> None:

    rc('font', ** {
       'family': 'serif',
       'serif': ["Times"]
       })
    rc('text', usetex=True)

    print(OmegaConf.to_yaml(cfg))

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)

    # Set device
    use_cuda = not cfg.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Get loaders
    _, test_loader, _ = init_dataset(cfg)

    # Get model and hooks
    model = init_classifier(cfg).to(device)

    model = SpecificLayerTypeOutputExtractor_wrapper(model, layer_type=torch.nn.ReLU)

    # Load Classifier
    classifier_filepath = classifier_ckpt_namer(model_name=model.name, cfg=cfg)
    model.load_state_dict(torch.load(classifier_filepath))

    # Get images
    imgs, _ = test_loader.__iter__().__next__()
    imgs = imgs.to(device)

    # Forward run
    _ = model(imgs)

    # breakpoint()

    layer_names = list(model.layer_outputs.keys())

    for layer_id in layer_names:
        number_of_images = 10
        for i in tqdm(range(number_of_images)):
            plot_activations(model.layer_outputs[layer_id][i].detach().cpu(), filepath=cfg.directory +
                            f"figs/activations/{classifier_filepath.split('/')[-1][:-3]}/{layer_id}/", file_name=f"image_{i}")
            
            

    


if __name__ == "__main__":
    main()
