#!/bin/bash

# Adam optimizer
mkdir "log"
python main.py --arch ETCModel --model-name DWT_Tiny_Half_R18_Hinton_CIFAR \
--data-name CIFAR10 --initial-epoch 0 --select-gpu 0 --test True \
>./log/DWT_Tiny_Half_R18_Hinton_CIFAR0.txt &

python main.py --arch ETCModel --model-name DWT_Tiny_Half_R18_Hinton_CIFAR \
--data-name CIFAR10 --initial-epoch 0 --select-gpu 1 --test True \
>./log/DWT_Tiny_Half_R18_Hinton_CIFAR1.txt &

python main.py --arch ETCModel --model-name DWT_Tiny_Half_R18_Hinton_CIFAR \
--data-name CIFAR10 --initial-epoch 0 --select-gpu 2 --test True \
>./log/DWT_Tiny_Half_R18_Hinton_CIFAR2.txt &

python main.py --arch ETCModel --model-name DWT_Tiny_Half_R18_Hinton_CIFAR \
--data-name CIFAR10 --initial-epoch 0 --select-gpu 3 --test True \
>./log/DWT_Tiny_Half_R18_Hinton_CIFAR3.txt &

python main.py --arch ETCModel --model-name DWT_Tiny_Half_R18_Hinton_CIFAR \
--data-name CIFAR10 --initial-epoch 0 --select-gpu 4 --test True \
>./log/DWT_Tiny_Half_R18_Hinton_CIFAR4.txt &

python main.py --arch ETCModel --model-name DWT_Tiny_Half_R18_Efficient_CIFAR \
--data-name CIFAR10 --initial-epoch 0 --select-gpu 5 --test True \
>./log/DWT_Tiny_Half_R18_Efficient_CIFAR5.txt &

python main.py --arch ETCModel --model-name DWT_Tiny_Half_R18_Efficient_CIFAR \
--data-name CIFAR10 --initial-epoch 0 --select-gpu 6 --test True \
>./log/DWT_Tiny_Half_R18_Efficient_CIFAR6.txt &

python main.py --arch ETCModel --model-name DWT_Tiny_Half_R18_Efficient_CIFAR \
--data-name CIFAR10 --initial-epoch 0 --select-gpu 7 --test True \
>./log/DWT_Tiny_Half_R18_Efficient_CIFAR7.txt &

python main.py --arch ETCModel --model-name DWT_Tiny_Half_R18_Efficient_CIFAR \
--data-name CIFAR10 --initial-epoch 0 --select-gpu 8 --test True \
>./log/DWT_Tiny_Half_R18_Efficient_CIFAR8.txt &

python main.py --arch ETCModel --model-name DWT_Tiny_Half_R18_Efficient_CIFAR \
--data-name CIFAR10 --initial-epoch 0 --select-gpu 9 --test True \
>./log/DWT_Tiny_Half_R18_Efficient_CIFAR9.txt &

echo "[INFO]Starting!"
