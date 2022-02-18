"""
"""

import numpy as np
from omegaconf import DictConfig, OmegaConf
import hydra
import time
from os.path import join
from matplotlib import rc
import matplotlib.pyplot as plt

# PYTORCH UTILS
from pytorch_utils.visualization import neuron_visualizer
from mpl_utils import plot_tensors


# Initializers
from .init import *

from .utils.namers import classifier_ckpt_namer


@hydra.main(config_path="/home/metehan/icip/src/configs", config_name="mnist")
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

    use_cuda = not cfg.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    train_loader, test_loader, data_params = init_dataset(cfg)

    model = init_classifier(cfg).to(device)

    # classifier_filepath = classifier_ckpt_namer(model_name=cfg.nn.classifier, cfg=cfg)
    # model.load_state_dict(torch.load(
    #     cfg.directory + f"checkpoints/classifiers/{cfg.dataset.name}/"+"/LeNet_adam_none_0.0010_none_ep_40.pt"))

    classifier_filepath = classifier_ckpt_namer(model_name=model.name, cfg=cfg, multi_phase=True)
    model.load_state_dict(torch.load(classifier_filepath))

    layer_name = "conv2"
    images, visualizations = neuron_visualizer(
        model=model, layer_name=layer_name, data_loader=test_loader, starting_image="max")

    # breakpoint()
    plot_tensors(tensor_list=images, filepath=cfg.directory+"figs/neurons/",
                 file_name=model.name+"_starting_images", title="Starting Images")
    plot_tensors(tensor_list=visualizations, filepath=cfg.directory+"figs/neurons/",
                 file_name=model.name+"_"+layer_name, title="Filter Maximizers")


if __name__ == "__main__":
    main()
