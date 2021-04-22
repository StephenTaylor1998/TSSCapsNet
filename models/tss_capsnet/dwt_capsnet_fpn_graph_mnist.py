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

import numpy as np
import tensorflow as tf
from models.layers import RoutingA
from models.layers.layers_efficient import PrimaryCaps, Length, Mask, generator_graph_mnist
from models.layers.operators_vector import Heterogeneous
from models.layers.routing_vector import RoutingVector
from models.layers.transform import DWT


def dwt_capsnet_graph(input_shape, num_classes=10, routing_name_list=None, regularize=1e-4, name=None):
    """
    reimplement for cifar dataset

    """
    inputs = tf.keras.Input(input_shape)
    # (28, 28, 1) ==>> (26, 26, 32)
    x = tf.keras.layers.Conv2D(32, 3, activation="relu", padding='valid', kernel_initializer='he_normal')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    # (26, 26, 32) ==>> (24, 24, 32)
    x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    # (24, 24, 32) ==>> (22, 22, 64)
    x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    # (22, 22, 64) ==>> (20, 20, 64)
    x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    # (20, 20, 64) ==>> (18, 18, 32)
    x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    # (18, 18, 32) ==>> (9, 9, 128)
    x = DWT()(x)
    x = PrimaryCaps(128, x.shape[1], 16, 8)(x)

    digit_caps = RoutingVector(num_classes, routing_name_list, regularize)(x)

    # x = layers.LayerNormalization()(x)
    # digit_caps = FCCaps(10, 16)(x)

    digit_caps_len = Length(name='length_capsnet_output')(digit_caps)

    digit_caps_len = Heterogeneous(num_class=10)((x, digit_caps_len))

    # digit_caps_len = tf.keras.layers.Softmax()(digit_caps_len)

    return tf.keras.Model(inputs=inputs, outputs=[digit_caps, digit_caps_len], name=name)


def build_graph(input_shape, mode, num_classes, routing_name_list, regularize=1e-4, name=''):
    inputs = tf.keras.Input(input_shape)
    y_true = tf.keras.layers.Input(shape=(10,))
    noise = tf.keras.layers.Input(shape=(10, 16))

    efficient_capsnet = dwt_capsnet_graph(input_shape, num_classes, routing_name_list, regularize, name)
    efficient_capsnet.summary()

    digit_caps, digit_caps_len = efficient_capsnet(inputs)
    noised_digitcaps = tf.keras.layers.Add()([digit_caps, noise])  # only if mode is play

    masked_by_y = Mask()([digit_caps, y_true])
    masked = Mask()(digit_caps)
    masked_noised_y = Mask()([noised_digitcaps, y_true])

    generator = generator_graph_mnist(input_shape)


    generator.summary()


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
