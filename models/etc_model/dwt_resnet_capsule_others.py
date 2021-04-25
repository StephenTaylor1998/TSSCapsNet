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

import tensorflow as tf
from tensorflow.keras import layers
from models.layers.layers_hinton import DigitCaps
from models.layers.layers_efficient import PrimaryCaps, Length, FCCaps
from models.etc_model.resnet_cifar_dwt import build_graph as build_resnet_dwt_backbone


def capsnet_graph(input_shape, num_classes, routing_name, depth=18):

    inputs = tf.keras.Input(input_shape)
    # (32, 32, 3) ==>> (8, 8, 128)
    x = build_resnet_dwt_backbone(input_shape, num_classes, depth, tiny=True, half=True, backbone=True)(inputs)

    x = layers.BatchNormalization()(x)
    # (4, 4, 256) ==>> (1, 1, 256) ==>> (32, 8)
    x = PrimaryCaps(256, x.shape[1], 32, 8)(x)
    if routing_name == "Hinton":
        x = tf.reshape(x, (-1, 1, 1, 32, 8))
        digit_caps = DigitCaps(10, 16, routing=3)(x)
    elif routing_name == "Efficient":
        digit_caps = FCCaps(10, 16)(x)
    else:
        raise NotImplementedError
    digit_caps_len = Length()(digit_caps)
    # digit_caps_len = Heterogeneous(num_class=10)((x, digit_caps_len))
    return tf.keras.Model(inputs=[inputs], outputs=[digit_caps_len])


def build_graph(input_shape, num_classes, routing_name, depth=18):
    efficient_capsnet = capsnet_graph(input_shape, num_classes, routing_name, depth)
    efficient_capsnet.summary()
    return efficient_capsnet
