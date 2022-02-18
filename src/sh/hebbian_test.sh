#!/bin/bash 

export CUDA_VISIBLE_DEVICES="1"
export PYTHONPATH="/home/metehan/icip/src/lib/"

# COMMAND="python -m src.attack_mnist --multirun train.reg.active=[hebbian,l1_conv2] train.reg.hebbian.tobe_regularized=['relu2'] nn.implicit_normalization=l2 nn.normalize_input=false train.reg.hebbian.scale=0.1 train.reg.hebbian.lamda=0.1 train.reg.hebbian.k=10 nn.divisive.sigma=0.5 nn.thresholding=[0.8,0.8],[0.8,1.0],[0.8,1.2],[1.0,0.8],[1.0,1.0]"
# echo $COMMAND
# eval $COMMAND


COMMAND="python -m src.hebbian_test train.type=standard nn.classifier=Standard_Nobias_VGG train.reg.active=none train.epochs=100 nn.lr=0.001"
echo $COMMAND
eval $COMMAND

# COMMAND="python -m src.attack_cifar nn.classifier=Standard_VGG train.epochs=100 train.type=noisy train.reg.active=none attack.epsilon=0.0156862745 attack.step_size=0.00156862745"
# echo $COMMAND
# eval $COMMAND

# COMMAND="python -m src.attack_cifar nn.classifier=Standard_VGG train.epochs=100 train.type=noisy train.reg.active=none attack.epsilon=0.00784313725 attack.step_size=0.000784313725"
# echo $COMMAND
# eval $COMMAND

# COMMAND="python -m src.attack_cifar nn.classifier=Standard_VGG train.epochs=100 train.type=noisy train.reg.active=none attack.epsilon=0.00392156862 attack.step_size=0.000392156862"
# echo $COMMAND
# eval $COMMAND
