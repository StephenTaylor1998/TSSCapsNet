import argparse


def get_terminal_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default="EfficientCapsNet",
                        choices=["EfficientCapsNet", "CapsNet", "TSSCapsNet", "TSSEfficientCapsNet"],
                        help='MODEL ARCH (default: "EfficientCapsNet")')
    parser.add_argument('--model-name', type=str, default="Efficient_CapsNet",
                        choices=["DCT_CapsNet_Attention", "DCT_CapsNet_GumbelGate", "DCT_CapsNet",
                                 "DCT_Efficient_CapsNet", "RFFT_Efficient_CapsNet", "Efficient_CapsNet",
                                 "CapsNet", "DWT_Efficient_CapsNet", "WST_Efficient_CapsNet"],
                        help='model name (default: DCT_CapsNet_Attention)')
    parser.add_argument('--data-name', type=str, default="MNIST",
                        choices=["MNIST", "MNIST_SHIFT", "FASHION_MNIST", "FASHION_MNIST_SHIFT",
                                 "CIFAR10", "CIFAR10_SHIFT"],
                        help='dataset name (default: MNIST)')
    parser.add_argument('--initial-epoch', type=int, default=0,
                        help='initial epoch (default: "0")')
    parser.add_argument('--select-gpu', type=int, default=None,
                        help='select gpu (default: "0")')
    parser.add_argument('--test', type=bool, default=True,
                        help='test model after training (default: "True")')
    return parser.parse_args()
