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


# ATTACK CODES
from deepillusion.torchattacks import FGSM, RFGSM, PGD
from deepillusion.torchattacks.analysis import whitebox_test

# PYTORCH UTILS
from pytorch_utils.surgery import LayerOutputExtractor_wrapper, SpecificLayerTypeOutputExtractor_wrapper
from pytorch_utils.analysis import count_parameter
from pytorch_utils.test import test_noisy, get_noisy_images
from pytorch_utils.layers import DivisiveNormalization2d, AdaptiveThreshold

# MPL utils
from mpl_utils import save_gif, plot_images

# Initializers
from .init import *

from .utils.train_test import standard_epoch, standard_test
from .utils.namers import classifier_ckpt_namer, classifier_params_string
from .utils.parsers import parse_regularizer


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
    # breakpoint()
    data, _ = test_loader.__iter__().__next__()
    noise_std = 0.0
    noisy_data = get_noisy_images(data, noise_std)
    noisy_data = noisy_data.permute(0,2,3,1).numpy()
    plot_images(noisy_data, filepath=cfg.directory + "figs/noisy_data/", file_name=str(noise_std)+".pdf")
   


if __name__ == "__main__":
    main()
