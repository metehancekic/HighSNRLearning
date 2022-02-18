"""
Example Run
python -m src.cifar.main  --model VGG11 -tr -sm
"""

from cgi import test
import numpy as np
from omegaconf import DictConfig, OmegaConf
import hydra
import time
import matplotlib.pyplot as plt
from matplotlib import rcParams
from tqdm import tqdm
import torch


# ATTACK CODES
from deepillusion.torchattacks import FGSM, RFGSM, PGD
from deepillusion.torchattacks.analysis import whitebox_test

# PYTORCH UTILS
from pytorch_utils.surgery import LayerOutputExtractor_wrapper, SpecificLayerTypeOutputExtractor_wrapper
from pytorch_utils.analysis import count_parameter
from pytorch_utils.test import test_noisy
from pytorch_utils.layers import DivisiveNormalization2d, AdaptiveThreshold

# MPL utils
from mpl_utils import save_gif

# Initializers
from .init import *

from .utils.train_test import single_epoch, standard_test
from .utils.namers import classifier_ckpt_namer, classifier_params_string
from .utils.parsers import parse_regularizer
from .models.custom_models import topk_LeNet, topk_VGG, T_LeNet, TT_LeNet, Leaky_LeNet, BT_LeNet, NT_LeNet, Nl1T_LeNet, Tdn_LeNet, Dn_LeNet, Custom_LeNet, ThresholdingLeNet, Implicit_VGG, Implicit_Divisive_VGG, Divisive_VGG, Standard_VGG, Implicit_Divisive_Threshold_VGG, Implicit_Divisive_Adaptive_Threshold_VGG, Standard_Nobias_VGG



@hydra.main(config_path="/home/metehan/icip/src/configs", config_name="cifar")
def main(cfg: DictConfig) -> None:

    rcParams["font.family"] = "sans-serif"
    print(OmegaConf.to_yaml(cfg))

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)

    use_cuda = not cfg.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    
    train_loader, test_loader, data_params = init_dataset(cfg)
    model = init_classifier(cfg).to(device)

    logger = init_logger(cfg, model.name)

    if not cfg.no_tensorboard:
        writer = init_tensorboard(cfg, model.name)
    else:
        writer = None

    logger.info(model)

    optimizer, scheduler = init_optimizer_scheduler(
        cfg, model, len(train_loader), printer=logger.info, verbose=True)

    _ = count_parameter(model=model, logger=logger.info, verbose=True)

    # classifier_filepath = f"/home/metehan/hebbian/checkpoints/classifiers/mnist/multi_phase/threshold_[0.8, 0.0]_multiple_phases_0.1Custom_LeNet_div_0.5_thr_[0.8, 0.0]_l2_adam_none_0.0010_hebbian_0.1_10_0.1_1_['relu2']_ep_40.pt"
    classifier_filepath = classifier_ckpt_namer(model_name=model.name, cfg=cfg)
    # classifier_filepath = "/home/metehan/hebbian/checkpoints/classifiers/CIFAR10/Implicit_Divisive_VGG16_div_0.1_adaptive_threshold_1.0_adam_step_0.0100_l1_0.001_hebbian_0.001_4_0.1_0.1_1_['relu1']_ep_100.pt"
    model.load_state_dict(torch.load(classifier_filepath))

    # model_base = Standard_VGG().to(device)
    # classifier_filepath = "/home/metehan/hebbian/checkpoints/classifiers/CIFAR10/VGG16_adam_step_0.0010_ep_50.pt"
    classifier_filepath = "/home/metehan/icip/checkpoints/classifiers/CIFAR10/Nobias_VGG16_adam_step_0.0010_ep_100.pt"
    model_base = Standard_Nobias_VGG().to(device)
    model_base.load_state_dict(torch.load(classifier_filepath))
    
    classifier_filepath = "/home/metehan/icip/checkpoints/classifiers/CIFAR10/Nobias_VGG16_adam_step_0.0010_advtr_PGD_inf_0.0314_0.04_0.00784_7_True_1_cross_entropy_ep_100.pt"
    model_adv = Standard_Nobias_VGG().to(device)
    model_adv.load_state_dict(torch.load(classifier_filepath))


    model = SpecificLayerTypeOutputExtractor_wrapper(model, layer_type=torch.nn.Conv2d)
    model_base = SpecificLayerTypeOutputExtractor_wrapper(model_base, layer_type=torch.nn.Conv2d)
    model_adv = SpecificLayerTypeOutputExtractor_wrapper(model_adv, layer_type=torch.nn.Conv2d)
    
    data, target = test_loader.__iter__().__next__()
    data, target = data.to(device), target.to(device)

    _ = model(data)
    _ = model_base(data)
    _ = model_adv(data)

    # breakpoint()

    act = model.layer_outputs["features.0"].detach().cpu().numpy()
    act_base = model_base.layer_outputs["features.0"].detach().cpu().numpy()
    act_adv = model_adv.layer_outputs["features.0"].detach().cpu().numpy()


    max1 = np.abs(act).max()
    max2 = np.abs(act_base).max()
    max3 = np.abs(act_adv).max()

    abs_max = max(max1, max2, max3)

    xlims = (-abs_max -0.0001, abs_max+0.0001)

    bin_edges = np.linspace(*xlims, 50)

    hist, _ = np.histogram(act, bin_edges, density=True)

    color, edgecolor = ("orange", "darkorange")

    plt.bar(
        bin_edges[:-1] + np.diff(bin_edges) / 2,
        hist,
        width=(bin_edges[1] - bin_edges[0]),
        alpha=0.5,
        edgecolor="none",
        color=color,
        )
    plt.step(
        np.array([*bin_edges, bin_edges[-1] + (bin_edges[1] - bin_edges[0])]),
        np.array([0, *hist, 0]),
        label=f"Ours",
        where="pre",
        color=edgecolor,
        )
    ax = plt.gca()
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.grid()

    hist, _ = np.histogram(act_base, bin_edges, density=True)
    color, edgecolor = ("cyan", "darkcyan")

    plt.bar(
        bin_edges[:-1] + np.diff(bin_edges) / 2,
        hist,
        width=(bin_edges[1] - bin_edges[0]),
        alpha=0.5,
        edgecolor="none",
        color=color,
        )
    plt.step(
        np.array([*bin_edges, bin_edges[-1] + (bin_edges[1] - bin_edges[0])]),
        np.array([0, *hist, 0]),
        label=f"Base",
        where="pre",
        color=edgecolor,
        )
    ax = plt.gca()
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.grid()

    hist, _ = np.histogram(act_adv, bin_edges, density=True)

    color, edgecolor = ("chocolate", "saddlebrown")

    plt.bar(
        bin_edges[:-1] + np.diff(bin_edges) / 2,
        hist,
        width=(bin_edges[1] - bin_edges[0]),
        alpha=0.5,
        edgecolor="none",
        color=color,
        )
    plt.step(
        np.array([*bin_edges, bin_edges[-1] + (bin_edges[1] - bin_edges[0])]),
        np.array([0, *hist, 0]),
        label=f"Adversarial",
        where="pre",
        color=edgecolor,
        )
    ax = plt.gca()
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.grid()

    plt.legend()

    filepath = cfg.directory + f"figs/neuron_regime/"
    if not os.path.exists(os.path.dirname(filepath+f"{classifier_params_string(model.name, cfg)}/")):
        os.makedirs(os.path.dirname(filepath+f"{classifier_params_string(model.name, cfg)}/"))
    plt.savefig(filepath + f"{classifier_params_string(model.name, cfg)}/" + f"regime.pdf")
    plt.close()





if __name__ == "__main__":
    main()
