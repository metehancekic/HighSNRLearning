"""
Example Run
python -m src.cifar.main  --model VGG11 -tr -sm
"""

import numpy as np
from omegaconf import DictConfig, OmegaConf
import hydra
import time

# ATTACK CODES
from deepillusion.torchattacks import FGSM, RFGSM, PGD, PGD_EOT

# PYTORCH UTILS
from pytorch_utils.surgery import LayerOutputExtractor_wrapper
from pytorch_utils.analysis import count_parameter

# MPL utils
from mpl_utils import save_gif

# Initializers
from .init import *

from .utils.train_test import standard_epoch, standard_test
from .utils.namers import classifier_ckpt_namer, classifier_params_string
from .utils.parsers import parse_regularizer
from .kpp import KMeansPPInitializer, extract_patches


@hydra.main(config_path="/home/metehan/icip/src/configs", config_name="mnist")
def main(cfg: DictConfig) -> None:

    import warnings
    warnings.filterwarnings("ignore")

    print(OmegaConf.to_yaml(cfg))

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)

    use_cuda = not cfg.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    train_loader, test_loader, data_params = init_dataset(cfg)
    model = init_classifier(cfg).to(device)

    model = LayerOutputExtractor_wrapper(model, layer_names=["relu1", "relu2"])
    logger = init_logger(cfg, model.name)

    if not cfg.no_tensorboard:
        writer = init_tensorboard(cfg, model.name)
    else:
        writer = None

    logger.info(model)


    dataset = torch.tensor(train_loader.dataset.data, dtype=torch.float)
    dataset /= dataset.max()

    patches = extract_patches(images=dataset.unsqueeze(1), patch_shape=(5, 5, 1), stride=1)

    patches_l1_norms = torch.abs(patches).sum(dim=(1, 2, 3))

    patches = patches[patches_l1_norms > 1.0]

    KMeansPPInitializer(model.conv1.weight, patches.cpu().numpy())

    optimizer, scheduler = init_optimizer_scheduler(
        cfg, model, len(train_loader), printer=logger.info, verbose=True)

    _ = count_parameter(model=model, logger=logger.info, verbose=True)

    weight_list = [None]*(cfg.train.epochs+1)
    weight_list[0] = model.conv1.weight.detach().cpu()

    weight_list2 = [None]*(cfg.train.epochs+1)
    weight_list2[0] = model.conv2.weight.detach().cpu()[:, 1:2, :, :]
    for epoch in range(1, cfg.train.epochs+1):
        start_time = time.time()

        tr_loss, tr_acc = standard_epoch(cfg=cfg, model=model, train_loader=train_loader,
                                         optimizer=optimizer, scheduler=scheduler, verbose=False)
        end_time = time.time()

        logger.info(f'{epoch} \t {end_time - start_time:.0f} \t {tr_loss:.4f} \t {tr_acc:.4f}')

        if epoch % cfg.log_interval == 0 or epoch == cfg.train.epochs:
            test_loss, test_acc = standard_test(
                model=model, test_loader=test_loader, verbose=False, progress_bar=False)
            logger.info(f'Test  \t loss: {test_loss:.4f} \t acc: {test_acc:.4f}')

        weight_list[epoch] = model.conv1.weight.detach().cpu()
        weight_list2[epoch] = model.conv2.weight.detach().cpu()[:, 1:2, :, :]

    # Save checkpoint
    if cfg.save_model:
        os.makedirs(cfg.directory + "checkpoints/classifiers/", exist_ok=True)
        classifier_filepath = classifier_ckpt_namer(model_name=model.name + "_kpp", cfg=cfg)
        torch.save(model.state_dict(), classifier_filepath)
        logger.info(f"Saved to {classifier_filepath}")
    
    
    save_gif(weight_list, filepath=cfg.directory + "gifs/first_layer/",
             file_name=classifier_params_string(model_name=model.name + "_kpp", cfg=cfg))
    save_gif(weight_list2, filepath=cfg.directory + "gifs/second_layer/",
             file_name=classifier_params_string(model_name=model.name + "_kpp", cfg=cfg))

    


if __name__ == "__main__":
    main()
