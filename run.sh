source venv/bin/activate &&

python demo.py --arch CapsNet --model-name CapsNet --data-name CIFAR10 \
--initial-epoch 0 --select-gpu 9 --test True


# arch
# ["EfficientCapsNet", "CapsNet", "TSSCapsNet", "TSSEfficientCapsNet"]

# model name
# ["DCT_CapsNet_Attention", "DCT_CapsNet_GumbelGate", "DCT_CapsNet",
# "DCT_Efficient_CapsNet", "RFFT_Efficient_CapsNet", "Efficient_CapsNet",
# "CapsNet", "DWT_Efficient_CapsNet", "WST_Efficient_CapsNet"],

# data name
# ["MNIST", "MNIST_SHIFT", "FASHION_MNIST", "FASHION_MNIST_SHIFT",
# "CIFAR10", "CIFAR10_SHIFT"]