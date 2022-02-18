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

# Initializers
from .init import *

from .utils import standard_test, test_noisy, SpecificLayerTypeOutputExtractor_wrapper
from .utils.namers import classifier_ckpt_namer, classifier_params_string
from .utils import Matching_VGG


@hydra.main(config_path="/home/metehan/StrongActivations/src/configs", config_name="cifar")
def main(cfg: DictConfig) -> None:

    rcParams["font.family"] = "sans-serif"
    print(OmegaConf.to_yaml(cfg))

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)

    use_cuda = not cfg.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    
    train_loader, test_loader, _ = init_dataset(cfg)
    model = init_classifier(cfg).to(device)

    logger = init_logger(cfg, model.name)


    logger.info(model)

    classifier_filepath = classifier_ckpt_namer(model_name=model.name, cfg=cfg)
    model.load_state_dict(torch.load(classifier_filepath))

    # model_base = Standard_VGG().to(device)
    # classifier_filepath = "/home/metehan/hebbian/checkpoints/classifiers/CIFAR10/VGG16_adam_step_0.0010_ep_50.pt"
    classifier_filepath = "/home/metehan/StrongActivations/checkpoints/CIFAR10/_VGG16_lr_0.0010_ep_100_seed_2022.pt"
    model_base = Matching_VGG().to(device)
    model_base.load_state_dict(torch.load(classifier_filepath))


    model = SpecificLayerTypeOutputExtractor_wrapper(model, layer_type=torch.nn.Conv2d)
    model_base = SpecificLayerTypeOutputExtractor_wrapper(model_base, layer_type=torch.nn.Conv2d)

    # Clean test

    test_loss, test_acc = standard_test(
        model=model, test_loader=test_loader, verbose=False, progress_bar=False)
    logger.info(f'Test  \t loss: {test_loss:.4f} \t acc: {test_acc:.4f}')

    noise_std = 0.1
    noisy_acc, noisy_loss, snrs = test_noisy(model, test_loader, noise_std)

    
    with open(cfg.directory + "logs/" + "noisy_cifar.txt", 'a') as file1:
        file1.write(f"Classifier: {classifier_filepath} \n")
        file1.write(f"Noise STD: {noise_std}")
        file1.write(f"\n")
        file1.write(f"Clean Loss: {test_loss}, Acc: {test_acc} \n")
        file1.write(f"Noisy Loss: {noisy_loss}, Acc: {noisy_acc} \n\n")

    layer_names = list(range(len(model.layer_inputs.keys())))

    # noise_levels_before_clip = np.logspace(np.log10(0.005), np.log10(0.2), 5)
    noise_levels_before_clip = np.arange(0.0, 0.20001, 0.01)

    SNRs = np.empty(len(layer_names))
            
    accuracy_list_ours = np.zeros(len(noise_levels_before_clip))
    accuracy_list_base = np.zeros(len(noise_levels_before_clip))
    for idx, level_before_clip in enumerate(tqdm(noise_levels_before_clip)):

        accuracy, _, layer_SNRs = test_noisy(model_base, test_loader, level_before_clip)
        accuracy_ours, _, layer_SNRs_ours = test_noisy(model, test_loader, level_before_clip)

        accuracy_list_ours[idx] = accuracy_ours
        accuracy_list_base[idx] = accuracy

        plt.plot(layer_names, layer_SNRs,
                    marker='o', label=f"Base: {(100*accuracy):.1f}", color="red")
        plt.plot(layer_names, layer_SNRs_ours,
                    marker='o', label=f"Ours: {(100*accuracy_ours):.1f}", color="green")

        plt.ylabel("SNR (dB)")
        plt.xlabel("Layers")
        plt.title(f"SNR vs Layers for $\sigma_0={{{level_before_clip}}}$")
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.grid()
        filepath = cfg.directory + f"figs/snr/"
        if not os.path.exists(os.path.dirname(filepath+f"{classifier_params_string(model.name, cfg)}/")):
            os.makedirs(os.path.dirname(filepath+f"{classifier_params_string(model.name, cfg)}/"))
        plt.savefig(filepath + f"{classifier_params_string(model.name, cfg)}/" + f"noise_{level_before_clip:.3f}.pdf")
        plt.close()

    plt.figure()
    plt.plot(noise_levels_before_clip, accuracy_list_ours, marker='o', label=f"Ours", color="green")
    plt.plot(noise_levels_before_clip, accuracy_list_base, marker='o', label=f"Base", color="red")
    plt.ylabel("Accuracy")
    plt.xlabel("Noise Variance")
    plt.title(f"Accuracy vs Input Noise")
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.grid()

    filepath = cfg.directory + f"figs/snr/"
    if not os.path.exists(os.path.dirname(filepath+f"{classifier_params_string(model.name, cfg)}/")):
        os.makedirs(os.path.dirname(filepath+f"{classifier_params_string(model.name, cfg)}/"))
    plt.savefig(filepath + f"{classifier_params_string(model.name, cfg)}/" + f"Accuracy.pdf")
    plt.close()


if __name__ == "__main__":
    main()
