import ray
import matplotlib.pyplot as plt
import torch
import os
from pygifsicle import optimize
import numpy as np
from typing import List


def plot_stats(train_loss, train_acc, test_loss, test_acc, l1_loss, filepath="") -> None:
    r"""
    Plot

    """

    num_epochs = len(train_loss)
    num_test_epochs = len(test_loss)
    epochs_per_test = num_epochs//num_test_epochs
    epochs_train = np.arange(1,num_epochs+1)
    epochs_test = np.arange(epochs_per_test,num_epochs+1,epochs_per_test)
    

    fig, ax1 = plt.subplots()
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('xent')
    ax1.plot(epochs_train, train_loss, color="tab:red", label="Train Loss")
    ax1.plot(epochs_test, test_loss, color="tab:blue", label="Test Loss")
    plt.legend()
    ax1.tick_params(axis='y')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    ax2.set_ylabel('l1')  # we already handled the x-label with ax1
    ax2.plot(epochs_train, l1_loss, color='green', label="l1 Loss")
    plt.grid()
    plt.legend()
    ax2.tick_params(axis='y')

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    os.makedirs(filepath, exist_ok=True)
    plt.savefig(filepath+"loss.pdf")


    fig, ax1 = plt.subplots()
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('Acc')
    ax1.plot(epochs_train, train_acc, color="tab:red", label="Train Acc")
    ax1.plot(epochs_test, test_acc, color="tab:blue", label="Test Acc")
    ax1.tick_params(axis='y')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.grid()
    plt.legend()
    os.makedirs(filepath, exist_ok=True)
    plt.savefig(filepath+"acc.pdf")

