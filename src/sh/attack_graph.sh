#!/bin/bash 

export CUDA_VISIBLE_DEVICES="2"
export PYTHONPATH="/home/metehan/icip/src/lib/"

# COMMAND="python -m src.attack_mnist --multirun train.reg.active=[hebbian,l1_conv2] train.reg.hebbian.tobe_regularized=['relu2'] nn.implicit_normalization=l2 nn.normalize_input=false train.reg.hebbian.scale=0.1 train.reg.hebbian.lamda=0.1 train.reg.hebbian.k=10 nn.divisive.sigma=0.5 nn.thresholding=[0.8,0.8],[0.8,1.0],[0.8,1.2],[1.0,0.8],[1.0,1.0]"
# echo $COMMAND
# eval $COMMAND


COMMAND="python -m src.attack_graph train.type=standard nn.classifier=Implicit_Divisive_Adaptive_Threshold_VGG nn.threshold=1.0 nn.divisive.sigma=0.0 nn.lr=0.001 train.epochs=100 train.reg.l1.scale=0.001 train.reg.hebbian.scale=0.1001 train.reg.active=['l1_weight','hebbian'] train.reg.hebbian.tobe_regularized=Conv2d"
echo $COMMAND
eval $COMMAND


