#!/bin/bash 

export CUDA_VISIBLE_DEVICES="0"
export PYTHONPATH="/home/metehan/hebbian/src/lib/"

# COMMAND="python -m src.plot_neuron_activators train.regularizer=none nn.classifier=LeNet"
# echo $COMMAND
# eval $COMMAND

# COMMAND="python -m src.plot_neuron_activators --multirun nn.implicit_normalization=l1,l2 nn.normalize_input=false,true"
# echo $COMMAND
# eval $COMMAND

# COMMAND="python -m src.plot_neuron_activators --multirun train.reg.active=['hebbian','l1_conv2','l1_conv1'] nn.implicit_normalization=l2 nn.normalize_input=false train.reg.hebbian.lamda=0.1 train.reg.hebbian.k=10 train.reg.hebbian.scale=0.1 nn.divisive.sigma=0.5 nn.thresholding=[0.8,0.8] train.reg.l1.scale=0.001,0.0001,0.00001 train.reg.hebbian.tobe_regularized=['relu2']"
# echo $COMMAND
# eval $COMMAND

COMMAND="python -m src.plot_neuron_activators train.reg.active=['none'] nn.implicit_normalization=none nn.normalize_input=false nn.classifier=LeNet"
echo $COMMAND
eval $COMMAND