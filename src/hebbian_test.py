"""
Example Run
python -m src.cifar.main  --model VGG11 -tr -sm
"""

from cProfile import label
import numpy as np
from omegaconf import DictConfig, OmegaConf
import hydra
import time
from torch.nn import Conv2d

import matplotlib.pyplot as plt

# ATTACK CODES
from deepillusion.torchattacks import FGSM, RFGSM, PGD
from deepillusion.torchattacks.analysis import whitebox_test

# PYTORCH UTILS
from pytorch_utils.surgery import LayerOutputExtractor_wrapper, SpecificLayerTypeOutputExtractor_wrapper
from pytorch_utils.analysis import count_parameter

# MPL utils
from mpl_utils import save_gif

# Initializers
from .init import *

from .utils.train_test import single_epoch, standard_test, hebbian_test
from .utils.namers import classifier_ckpt_namer, classifier_params_string
from .utils.parsers import parse_regularizer


@hydra.main(config_path="/home/metehan/icip/src/configs", config_name="cifar")
def main(cfg: DictConfig) -> None:

    print(OmegaConf.to_yaml(cfg))

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)

    use_cuda = not cfg.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    
    train_loader, test_loader, data_params = init_dataset(cfg)
    model = init_classifier(cfg).to(device)

    model = SpecificLayerTypeOutputExtractor_wrapper(
        model, layer_type=globals()[cfg.test.hebbian.layer])

    logger = init_logger(cfg, model.name)

    if not cfg.no_tensorboard:
        writer = init_tensorboard(cfg, model.name)
    else:
        writer = None

    logger.info(model)

    optimizer, scheduler = init_optimizer_scheduler(
        cfg, model, len(train_loader), printer=logger.info, verbose=True)

    _ = count_parameter(model=model, logger=logger.info, verbose=True)

    percent_list = [0.01,0.02,0.04,0.08]
    top_k_minus_bottom_dictionary = dict(standard=np.zeros((len(percent_list),13)),
                                         adversarial=np.zeros((len(percent_list),13)))
    for i in ["standard", "adversarial"]:
        cfg.train.type = i
        # classifier_filepath = f"/home/metehan/hebbian/checkpoints/classifiers/mnist/multi_phase/threshold_[0.8, 0.0]_multiple_phases_0.1Custom_LeNet_div_0.5_thr_[0.8, 0.0]_l2_adam_none_0.0010_hebbian_0.1_10_0.1_1_['relu2']_ep_40.pt"
        classifier_filepath = classifier_ckpt_namer(model_name=model.name, cfg=cfg)
        model.load_state_dict(torch.load(classifier_filepath))

        # Clean test
        for j, percent in enumerate(percent_list):
            cfg.test.hebbian.percent = percent
            test_loss, test_acc, top_k_minus_bottom = hebbian_test(cfg=cfg, model=model, test_loader=train_loader, verbose=False, progress_bar=False)
            top_k_minus_bottom_dictionary[i][j] = top_k_minus_bottom.cpu().detach().numpy()
            logger.info(f'Test  \t loss: {test_loss:.4f} \t acc: {test_acc:.4f}')
            logger.info(top_k_minus_bottom)
            logger.info(torch.mean(top_k_minus_bottom))

    colors = ["red", "blue", "yellow", "green", "black"]
    plt.figure()
    for i in range(4):
        plt.plot(np.arange(13), top_k_minus_bottom_dictionary["standard"][i,:], color=colors[i], linestyle='dashed', label=f"standard {percent_list[i]}")
        plt.plot(np.arange(13), top_k_minus_bottom_dictionary["adversarial"][i,:], color=colors[i])
    plt.legend()
    plt.title("Top activations mean minus bottom activations means wrt percentage")
    plt.xlabel("Layer")
    plt.ylabel("top - bottom")
    plt.grid()
    plt.savefig("/home/metehan/icip/figs/hebbian/0.pdf")
    breakpoint()

    
if __name__ == "__main__":
    main()
