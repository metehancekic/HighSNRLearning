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


# PYTORCH UTILS
from pytorch_utils.surgery import LayerOutputExtractor_wrapper
from pytorch_utils.analysis import count_parameter

from mpl_utils.activations import plot_activations
from mpl_utils import plot_histograms

from .models.custom_models import Custom_LeNet
from .models import LeNet

# Initializers
from .init import *

from .utils.namers import classifier_ckpt_namer, classifier_params_string


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

    # Set device
    use_cuda = not cfg.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Get loaders
    _, test_loader, _ = init_dataset(cfg)

    # Get model and hooks
    # model = init_classifier(cfg).to(device)
    model = Custom_LeNet(normalize_input=False, implicit_normalization=cfg.nn.implicit_normalization, divisive_sigma=0.5, thresholding=[0.8, 0.8]).to(device)
    # model = LeNet().to(device)
    model = LayerOutputExtractor_wrapper(model, layer_names=["relu1", "relu2"])

    # Load Classifier
    # classifier_filepath = classifier_ckpt_namer(model_name=model.name, cfg=cfg)
    classifier_filepath = "/home/metehan/hebbian/checkpoints/classifiers/mnist/multi_phase/threshold_[0.8, 0.8]_Custom_LeNet_div_0.5_thr_[0.8, 0.8]_l2_adam_none_0.0010_hebbian_0.1_10_0.1_1_['relu2']_l1_conv2_0.0001_l1_conv1_0.0001_ep_40.pt"
    # classifier_filepath = "/home/metehan/hebbian/checkpoints/classifiers/mnist/multi_phase/threshold_[0.8, 0.8]_Custom_LeNet_div_0.5_thr_[0.8, 0.8]_l2_adam_none_0.0010_hebbian_0.1_10_0.1_1_['relu2']_l1_conv2_0.001_l1_conv1_0.001_ep_40.pt"
    # classifier_filepath = "/home/metehan/hebbian/checkpoints/classifiers/mnist/multi_phase/threshold_[0.8, 0.8]_Custom_LeNet_div_0.5_thr_[0.8, 0.8]_l2_adam_none_0.0010_hebbian_0.1_10_0.1_1_['relu2']_l1_conv2_1e-05_l1_conv1_1e-05_ep_40.pt"
    # classifier_filepath = "/home/metehan/hebbian/checkpoints/classifiers/mnist/LeNet_adam_none_0.0010_ep_40.pt"
    model.load_state_dict(torch.load(classifier_filepath))

    # Get images
    imgs, _ = test_loader.__iter__().__next__()
    imgs = imgs.to(device)

    # Forward run
    _ = model(imgs)

    # breakpoint()

    second_layer_activations = model.layer_outputs["relu2"].detach().cpu().permute(1,0,2,3).reshape(model.layer_outputs["relu2"].shape[1],-1)

    plot_histograms(second_layer_activations, filepath=cfg.directory +
                        f"figs/histograms_act/{classifier_filepath.split('/')[-1][:-3]}/", file_name=f"relu2")

    # Plot activations
    # number_of_images = 10
    # for i in tqdm(range(number_of_images)):
    #     plot_activations(model.layer_outputs["relu1"][i].detach().cpu(), filepath=cfg.directory +
    #                     f"figs/activations/{classifier_filepath.split('/')[-1][:-3]}/", file_name=f"relu1_{i}")
    #     plot_activations(model.layer_outputs["relu2"][i].detach().cpu(), filepath=cfg.directory +
    #                     f"figs/activations/{classifier_filepath.split('/')[-1][:-3]}/", file_name=f"relu2_{i}")

    


if __name__ == "__main__":
    main()
