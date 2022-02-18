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
from mpl_utils.bars import plot_bar

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
    model = init_classifier(cfg).to(device)
    model = LayerOutputExtractor_wrapper(model, layer_names=["relu1", "dn"])

    # Load Classifier
    classifier_filepath = classifier_ckpt_namer(model_name=model.name, cfg=cfg)
    model.load_state_dict(torch.load(classifier_filepath))

    # Get images
    imgs, _ = test_loader.__iter__().__next__()
    imgs = imgs.to(device)

    # Forward run
    _ = model(imgs)


    layer_outputs = model.layer_outputs["dn"].detach().cpu()
    layer_outputs = layer_outputs.permute(1,0,2,3)
    layer_outputs = layer_outputs.reshape(layer_outputs.shape[0],-1)
    
    
    
    x = np.arange(32)

    thresholds = [0.0, 0.1, 0.2, 0.4, 0.8, 1.6]
    for i in tqdm(thresholds):
        active_count = layer_outputs > i
        active_hist = torch.sum(active_count, dim=1)
        plot_bar(x, active_hist.detach().cpu(), filepath=cfg.directory +
                        f"figs/bars/{classifier_params_string(model_name=model.name, cfg=cfg)}/", file_name=f"threshold_{i}", title=f"Threshold {i}")


if __name__ == "__main__":
    main()
