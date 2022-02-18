"""
"""

import numpy as np
from omegaconf import DictConfig, OmegaConf
import hydra
import time
from os.path import join
from matplotlib import rc
import matplotlib.pyplot as plt

# ATTACK CODES
from deepillusion.torchattacks import FGSM, RFGSM, PGD, PGD_EOT

# PYTORCH UTILS
from pytorch_utils.surgery import LayerOutputExtractor_wrapper
from pytorch_utils.analysis import count_parameter

from mpl_utils.gif_creation import save_gif

# Initializers
from .init import *

from .utils.namers import classifier_ckpt_namer, classifier_params_string
from .models.custom_layers import LpConv2d
from .models import LeNet
from .models.custom_models import topk_LeNet, topk_VGG, T_LeNet, TT_LeNet, Leaky_LeNet, BT_LeNet, NT_LeNet, Nl1T_LeNet


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

    model = init_classifier(cfg)

    classifier_filepath = classifier_ckpt_namer(model_name=cfg.nn.classifier, cfg=cfg)
    model.load_state_dict(torch.load(classifier_filepath))

    save_gif([model.conv1.weight.detach()], filepath=cfg.directory + "figs/weights/",
             file_name=classifier_params_string(model_name=cfg.nn.classifier, cfg=cfg))


if __name__ == "__main__":
    main()
