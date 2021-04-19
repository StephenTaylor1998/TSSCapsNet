# Copyright 2021 Hang-Chi Shen. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

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
                            "CapsNet",
                            # Efficient_CapsNet
                            "Efficient_CapsNet",
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

                            "RFFT_E_MNIST",
                            "WST_E_MNIST",

                            # ETCModel
                            "RESNET18",
                            "RESNET34",
                            "RESNET50",
                            "RESNET18_Half",
                            "RESNET34_Half",
                            "RESNET50_Half",

                            "RESNET18_DWT",
                            "RESNET34_DWT",
                            "RESNET50_DWT",
                            "RESNET18_DWT_Tiny",
                            "RESNET34_DWT_Tiny",
                            "RESNET50_DWT_Tiny",
                            "RESNET18_DWT_Half",
                            "RESNET34_DWT_Half",
                            "RESNET50_DWT_Half",
                            "RESNET18_DWT_Tiny_Half",
                            "RESNET34_DWT_Tiny_Half",
                            "RESNET50_DWT_Tiny_Half",
                            "MOBILENETv2",

                            # ETCModel AND TSSCapsNet SUPPORT
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

                        ],
                        help='model name (default: DCT_CapsNet_Attention)')
    parser.add_argument('--data-name', type=str, default="MNIST",
                        choices=["MNIST", "MNIST_SHIFT", "FASHION_MNIST", "FASHION_MNIST_SHIFT",
                                 "CIFAR10", "CIFAR10_SHIFT", "SMALLNORB"],
                        help='dataset name (default: MNIST)')
    parser.add_argument('--initial-epoch', type=int, default=0,
                        help='initial epoch (default: "0")')
    parser.add_argument('--select-gpu', type=int, default=None,
                        help='select gpu (default: "0")')
    parser.add_argument('--test', type=bool, default=True,
                        help='test model after training (default: "True")')
    parser.add_argument('--heterogeneous', type=bool, default=False,
                        help='Add heterogeneous component to model (default: "False")')
    parser.add_argument('--optimizer', type=str, default="Adam",
                        choices=["Adam", "SGD"],
                        help='optimizer (default: Adam)')
    return parser.parse_args()
