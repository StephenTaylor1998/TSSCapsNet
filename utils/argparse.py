import argparse


def get_terminal_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default="EfficientCapsNet",
                        choices=["CapsNet",
                                 "TSSCapsNet",
                                 "EfficientCapsNet",
                                 "TSSEfficientCapsNet",
                                 "ETCModel"],
                        help='MODEL ARCH (default: "EfficientCapsNet")')
    parser.add_argument('--model-name', type=str, default="Efficient_CapsNet",
                        choices=[
                            # CapsNet
                            "CapsNet"
                            # TSSCapsNet
                            "DCT_CapsNet",
                            "DCT_CapsNet_GumbelGate",
                            "DCT_CapsNet_Attention"
                            # TSSCapsNet
                            "DCT_E_MNIST",
                            "DCT_H_A_MNIST",
                            "DCT_H_MNIST",
                            "DCT_H_Gumbel_MNIST",
                            "DWT_E_MNIST",
                            "DWT_E_SMALLNORB",
                            "DWT_FPN_MNIST",
                            "DWT_Tiny_FPN_MNIST",
                            "DWT_Attention_FPN_MNIST",
                            "DWT_FPN_SMALLNORB",
                            "DWT_Tiny_FPN_SMALLNORB",
                            "DWT_Attention_FPN_SMALLNORB",

                            "DWT_Tiny_Half_R18_Tiny_FPN_CIFAR",
                            "DWT_Tiny_Half_R34_Tiny_FPN_CIFAR",
                            "DWT_Tiny_Half_R50_Tiny_FPN_CIFAR",
                            "DWT_Half_R18_Tiny_FPN_CIFAR",
                            "DWT_Half_R34_Tiny_FPN_CIFAR",
                            "DWT_Half_R50_Tiny_FPN_CIFAR",
                            "DWT_Tiny_R18_Tiny_FPN_CIFAR",
                            "DWT_Tiny_R34_Tiny_FPN_CIFAR",
                            "DWT_Tiny_R50_Tiny_FPN_CIFAR",

                            "DWT_Tiny_Half_R18_FPN_CIFAR",
                            "DWT_Tiny_Half_R34_FPN_CIFAR",
                            "DWT_Tiny_Half_R50_FPN_CIFAR",
                            "DWT_Half_R18_FPN_CIFAR",
                            "DWT_Half_R34_FPN_CIFAR",
                            "DWT_Half_R50_FPN_CIFAR",
                            "DWT_Tiny_R18_FPN_CIFAR",
                            "DWT_Tiny_R34_FPN_CIFAR",
                            "DWT_Tiny_R50_FPN_CIFAR",

                            "DWT_Tiny_Half_R18_Attention_FPN_CIFAR",
                            "DWT_Tiny_Half_R34_Attention_FPN_CIFAR",
                            "DWT_Tiny_Half_R50_Attention_FPN_CIFAR",
                            "DWT_Half_R18_Attention_FPN_CIFAR",
                            "DWT_Half_R34_Attention_FPN_CIFAR",
                            "DWT_Half_R50_Attention_FPN_CIFAR",
                            "DWT_Tiny_R18_Attention_FPN_CIFAR",
                            "DWT_Tiny_R34_Attention_FPN_CIFAR",
                            "DWT_Tiny_R50_Attention_FPN_CIFAR",

                            "RFFT_E_MNIST",
                            "WST_E_MNIST"

                            # ETCModel
                            "RESNET18",
                            "RESNET34",
                            "RESNET50",
                            "RESNET_DWT18",
                            "RESNET_DWT34",
                            "RESNET_DWT50",
                            "GHOSTNET",
                            "MOBILENETv2",
                            "DWT_Caps_FPN",
                            "DWT_Caps_FPNTiny",
                            "DWT_Caps_Attention",
                            "CapsNet_Without_Decoder",
                        ],
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
