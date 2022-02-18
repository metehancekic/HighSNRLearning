"""
Example Run
python -m src.cifar.main  --model VGG11 -tr -sm
"""

import numpy as np
from omegaconf import DictConfig, OmegaConf
import hydra
import time
from tqdm import tqdm
from matplotlib import pyplot as plt

# ATTACK CODES
from deepillusion.torchattacks import FGSM, RFGSM, PGD
from deepillusion.torchattacks.analysis import whitebox_test

# PYTORCH UTILS
from pytorch_utils.surgery import LayerOutputExtractor_wrapper
from pytorch_utils.analysis import count_parameter

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
    model.load_state_dict(torch.load(classifier_filepath))

    # Clean test

    test_loss, test_acc = standard_test(
        model=model, test_loader=test_loader, verbose=False, progress_bar=False)
    logger.info(f'Test  \t loss: {test_loss:.4f} \t acc: {test_acc:.4f}')

    #--------------------------------------------------#
    #------------ Adversarial Argumenrs ---------------#
    #--------------------------------------------------#
    classifier_filepath = "/home/metehan/icip/checkpoints/classifiers/CIFAR10/Nobias_VGG16_adam_step_0.0010_ep_100.pt"
    model_base = Standard_Nobias_VGG().to(device)
    model_base.load_state_dict(torch.load(classifier_filepath))

    acc_ours = np.zeros(6)
    acc_base = np.zeros(6)
    loss_ours = np.zeros(6)
    loss_base = np.zeros(6)

    loss_ours[0], acc_ours[0] = standard_test(
        model=model, test_loader=test_loader, verbose=False, progress_bar=False)
    
    loss_base[0], acc_base[0] = standard_test(
        model=model_base, test_loader=test_loader, verbose=False, progress_bar=False)
    
    for idx, eps in tqdm(enumerate(np.array([1.0,2.0,3.0,4.0,5.0])/255)):    

        attack_params = {
            "norm": "inf",
            "eps": eps,
            "alpha": 0.04,
            "step_size": eps/10,
            "num_steps": 30,
            "random_start": False,
            "num_restarts": 1,
            "EOT_size": 1,
            }

        loss_function = "cross_entropy"

        adversarial_args = dict(
            attack=PGD,
            attack_args=dict(
                net=model, data_params=data_params, attack_params=attack_params, loss_function=loss_function
                ),
            )

        loss_ours[idx+1], acc_ours[idx+1] = whitebox_test(model=model, test_loader=test_loader,
                                        adversarial_args=adversarial_args, verbose=True, progress_bar=True)

        adversarial_args = dict(
            attack=PGD,
            attack_args=dict(
                net=model, data_params=data_params, attack_params=attack_params, loss_function=loss_function
                ),
            )
    
        loss_base[idx+1], acc_base[idx+1] = whitebox_test(model=model_base, test_loader=test_loader,
                                        adversarial_args=adversarial_args, verbose=True, progress_bar=True)

    with open(cfg.directory + "logs/" + "attacks_defended.txt", 'a') as file1:
        file1.write(f"Classifier: {classifier_filepath} \n")
        for key in adversarial_args["attack_args"]["attack_params"]:
            file1.write("\t" + key + ': ' +
                        str(adversarial_args["attack_args"]["attack_params"][key]))
        file1.write(f"\n")
        file1.write(f"Our Loss: {loss_ours}, Acc: {acc_ours} \n")
        file1.write(f"Base Loss: {loss_base}, Acc: {acc_base} \n")

    
    plt.figure()
    plt.plot(np.arange(0,6,1), acc_ours, marker='o', label=f"Ours", color="green")
    plt.plot(np.arange(0,6,1), acc_base, marker='o', label=f"Base", color="red")
    plt.ylabel("Accuracy")
    plt.xlabel("Attack Budget (pixel)")
    plt.title(f"Accuracy vs Attack Budget")
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.grid()

    filepath = cfg.directory + f"figs/adversarial/"
    if not os.path.exists(os.path.dirname(filepath+f"{classifier_params_string(model.name, cfg)}/")):
        os.makedirs(os.path.dirname(filepath+f"{classifier_params_string(model.name, cfg)}/"))
    plt.savefig(filepath + f"{classifier_params_string(model.name, cfg)}/" + f"Accuracy.pdf")
    plt.close()

    plt.figure()
    plt.plot(np.arange(0,6,1), loss_ours, marker='o', label=f"Ours", color="green")
    plt.plot(np.arange(0,6,1), loss_base, marker='o', label=f"Base", color="red")
    plt.ylabel("Loss")
    plt.xlabel("Attack Budget (pixel)")
    plt.title(f"Accuracy vs Attack Budget")
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.grid()

    filepath = cfg.directory + f"figs/adversarial/"
    if not os.path.exists(os.path.dirname(filepath+f"{classifier_params_string(model.name, cfg)}/")):
        os.makedirs(os.path.dirname(filepath+f"{classifier_params_string(model.name, cfg)}/"))
    plt.savefig(filepath + f"{classifier_params_string(model.name, cfg)}/" + f"Loss.pdf")
    plt.close()

    N = 6
    ind = np.arange(N) # the x locations for the groups
    width = 1.0
    fig = plt.figure()
    plt.bar(ind, acc_base*100, width, color='orange')
    plt.bar(ind, np.maximum(acc_ours*100-acc_base*100, np.zeros_like(acc_ours)), width, bottom=acc_base*100, color='yellowgreen')
    plt.ylabel('Accuracy')
    plt.title('Robustness against PGD attack')
    plt.xticks(ind, ('Clean', '1', '2', '3', '4', '5'))
    plt.yticks(np.arange(0, 101, 10))
    plt.legend(labels=['Base Robustness', 'Additional Robustness'])
    # plt.show()
    filepath = cfg.directory + f"figs/adversarial/"
    if not os.path.exists(os.path.dirname(filepath+f"{classifier_params_string(model.name, cfg)}/")):
        os.makedirs(os.path.dirname(filepath+f"{classifier_params_string(model.name, cfg)}/"))
    plt.savefig(filepath + f"{classifier_params_string(model.name, cfg)}/" + f"bar_acc.pdf")
    plt.close()

    N = 6
    ind = np.arange(N) # the x locations for the groups
    width = 0.35
    fig = plt.figure()
    plt.bar(ind, acc_base*100, width, color='orange')
    plt.bar(ind + 0.35, acc_ours*100, width, color='yellowgreen')
    plt.ylabel('Accuracy')
    plt.title('Robustness against PGD attack')
    plt.xticks(ind, ('Clean', '1', '2', '3', '4', '5'))
    plt.yticks(np.arange(0, 101, 10))
    plt.legend(labels=['Base', 'Ours'])
    # plt.show()
    filepath = cfg.directory + f"figs/adversarial/"
    if not os.path.exists(os.path.dirname(filepath+f"{classifier_params_string(model.name, cfg)}/")):
        os.makedirs(os.path.dirname(filepath+f"{classifier_params_string(model.name, cfg)}/"))
    plt.savefig(filepath + f"{classifier_params_string(model.name, cfg)}/" + f"bar2_acc.pdf")
    plt.close()

        


if __name__ == "__main__":
    main()
