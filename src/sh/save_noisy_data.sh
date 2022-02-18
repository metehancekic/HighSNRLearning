#!/bin/bash 

export CUDA_VISIBLE_DEVICES="0"
export PYTHONPATH="/home/metehan/hebbian/src/lib/"


COMMAND="python -m src.save_noisy_data"
echo $COMMAND
eval $COMMAND




