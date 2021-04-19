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
from models.layers.routing import Routing
from models.layers.operators import Heterogeneous
from models.etc_model.resnet_cifar_dwt import build_graph as build_resnet_dwt_backbone
from models.layers.layers_efficient import PrimaryCaps, Length, Mask, generator_graph_mnist


def dwt_capsnet_graph(input_shape, num_classes=10, routing_name_list=None,
                      regularize=1e-4, depth=18, tiny=True, half=True, name=None, heterogeneous=False):
    """
    reimplement for cifar dataset
    """
    inputs = tf.keras.Input(input_shape)
    # (32, 32, 3) ==>> (4, 4, 256)
    x = build_resnet_dwt_backbone(input_shape, num_classes, depth, tiny, half, backbone=True)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    # (4, 4, 256) ==>> (1, 1, 256) ==>> (32, 8)
    x = PrimaryCaps(256, x.shape[1], 32, 8)(x)

    digit_caps = Routing(num_classes, routing_name_list, regularize)(x)

    digit_caps_len = Length(name='length_capsnet_output')(digit_caps)

    if heterogeneous:
        digit_caps_len = Heterogeneous(num_class=num_classes)((x, digit_caps_len))

    return tf.keras.Model(inputs=inputs, outputs=[digit_caps, digit_caps_len], name=name)


def build_graph(input_shape, mode, num_classes, routing_name_list, regularize=1e-4,
                depth=18, tiny=True, half=True, name=None, heterogeneous=False):
    inputs = tf.keras.Input(input_shape)
    y_true = tf.keras.layers.Input(shape=(10,))
    noise = tf.keras.layers.Input(shape=(10, 16))

    efficient_capsnet = dwt_capsnet_graph(input_shape, num_classes, routing_name_list,
                                          regularize, depth, tiny, half, name, heterogeneous)

    efficient_capsnet.summary()
    print("\n\n")

    digit_caps, digit_caps_len = efficient_capsnet(inputs)
    noised_digitcaps = tf.keras.layers.Add()([digit_caps, noise])  # only if mode is play

    masked_by_y = Mask()([digit_caps, y_true])
    masked = Mask()(digit_caps)
    masked_noised_y = Mask()([noised_digitcaps, y_true])

    # generator = generator_graph_smallnorb(input_shape)
    generator = generator_graph_mnist(input_shape)

    generator.summary()
    print("\n\n")

    x_gen_train = generator(masked_by_y)
    x_gen_eval = generator(masked)
    x_gen_play = generator(masked_noised_y)

    if mode == 'train':
        return tf.keras.models.Model([inputs, y_true], [digit_caps_len, x_gen_train],
                                     name='DWT_Efficinet_CapsNet_Generator')
    elif mode == 'test':
        return tf.keras.models.Model(inputs, [digit_caps_len, x_gen_eval], name='DWT_Efficinet_CapsNet_Generator')
    elif mode == 'play':
        return tf.keras.models.Model([inputs, y_true, noise], [digit_caps_len, x_gen_play],
                                     name='DWT_Efficinet_CapsNet_Generator')
    else:
        raise RuntimeError('mode not recognized')
