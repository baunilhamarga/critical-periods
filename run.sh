#!/bin/bash  
  
# Navigate to the script directory  
# cd /path/to/your/script  
  
# Execute the Python script with different arguments
export CUDA_VISIBLE_DEVICES=0
./utils/shell_scripts.sh python no_aug.py
./utils/shell_scripts.sh python no_aug.py
./utils/shell_scripts.sh python no_aug.py
./utils/shell_scripts.sh python no_aug.py
export CUDA_VISIBLE_DEVICES=1
./utils/shell_scripts.sh python no_aug.py
./utils/shell_scripts.sh python no_aug.py
./utils/shell_scripts.sh python no_aug.py
./utils/shell_scripts.sh python no_aug.py

# ./shell_scripts.sh -m main
# ./utils/shell_scripts.sh imagenet50.py --dataset tiny_imagenet --architecture ResNet50
# ./shell_scripts.sh -m subsampling.annealing --architecture ResNet18 --dataset CIFAR100 --epoch 199
# ./shell_scripts.sh -m subsampling.annealing --architecture ResNet18 --dataset CIFAR100 --epoch 192
# ./shell_scripts.sh -m subsampling.annealing --architecture ResNet18 --dataset CIFAR100 --epoch 183
# ./shell_scripts.sh -m subsampling.annealing --architecture ResNet18 --dataset CIFAR100 --epoch 175
# ./shell_scripts.sh -m subsampling.annealing --architecture ResNet18 --dataset CIFAR100 --epoch 166
# ./shell_scripts.sh -m subsampling.annealing --architecture ResNet18 --dataset CIFAR100 --epoch 114
# ./shell_scripts.sh -m subsampling.annealing --architecture ResNet18 --dataset CIFAR100 --epoch 179
# [199, 192, 183, 175, 166, 114, 179]
