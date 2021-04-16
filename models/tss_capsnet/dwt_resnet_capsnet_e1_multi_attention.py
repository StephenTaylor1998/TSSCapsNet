# Copyright 2021 Vittorio Mazzia & Francesco Salvetti. All Rights Reserved.
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

import math
import numpy as np
import tensorflow as tf
from tensorflow import keras, Variable, float32

# from ..layers.transform.dwt import DWT
from models.layers import RoutingA
from models.layers.layers_efficient import PrimaryCaps, FCCaps, Length, Mask
from models.layers.operators import Heterogeneous
from tensorflow.keras import regularizers
from tensorflow.keras.regularizers import L2
from tensorflow.keras import layers, Sequential


class BasicBlock(layers.Layer):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.in_planes = in_planes
        self.planes = planes
        self.stride = stride
        self.conv1 = layers.Conv2D(self.planes, kernel_size=3, strides=self.stride, padding='same', use_bias=False,
                                   kernel_initializer='he_normal',
                                   kernel_regularizer=L2(1e-4)
                                   )
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(self.planes, kernel_size=3, strides=1, padding='same', use_bias=False,
                                   kernel_initializer='he_normal',
                                   kernel_regularizer=L2(1e-4)
                                   )
        self.bn2 = layers.BatchNormalization()
        self.relu = layers.ReLU()
        if self.stride != 1 or self.in_planes != self.expansion * self.planes:
            self.shortcut = Sequential([
                layers.Conv2D(self.expansion * self.planes, kernel_size=1, strides=self.stride, use_bias=False,
                              kernel_initializer='he_normal',
                              kernel_regularizer=L2(1e-4)
                              ),
                layers.BatchNormalization()
            ])
        else:
            self.shortcut = Sequential()

    def build(self, input_shape):
        self.built = True

    def get_config(self):
        config = {
            'in_planes': self.in_planes,
            'planes': self.planes,
            'stride': self.stride,
        }
        base_config = super(BasicBlock, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs, **kwargs):
        out = self.relu(self.bn1(self.conv1(inputs)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(inputs)
        out = self.relu(out)
        return out


class Bottleneck(layers.Layer):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.in_planes = in_planes
        self.planes = planes
        self.stride = stride

        self.conv1 = layers.Conv2D(self.planes, kernel_size=1, use_bias=False,
                                   kernel_initializer='he_normal',
                                   # kernel_regularizer=L2(1e-4)
                                   )
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(self.planes, kernel_size=3, strides=self.stride, padding='same', use_bias=False,
                                   kernel_initializer='he_normal',
                                   # kernel_regularizer=L2(1e-4)
                                   )
        self.bn2 = layers.BatchNormalization()
        self.conv3 = layers.Conv2D(self.expansion * self.planes, kernel_size=1, use_bias=False,
                                   kernel_initializer='he_normal',
                                   # kernel_regularizer=L2(1e-4)
                                   )
        self.bn3 = layers.BatchNormalization()
        self.relu = layers.ReLU()

        if self.stride != 1 or self.in_planes != self.expansion * self.planes:
            self.shortcut = Sequential([
                layers.Conv2D(self.expansion * self.planes, kernel_size=1, strides=self.stride, use_bias=False,
                              kernel_initializer='he_normal',
                              kernel_regularizer=L2(1e-4)
                              ),
                layers.BatchNormalization()
            ])
        else:
            self.shortcut = Sequential()

    def build(self, input_shape):
        self.built = True

    def get_config(self):
        config = {
            'in_planes': self.in_planes,
            'planes': self.planes,
            'stride': self.stride,
        }
        base_config = super(Bottleneck, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs, **kwargs):
        out = self.relu(self.bn1(self.conv1(inputs)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(inputs)
        out = self.relu(out)
        return out


class ResNetBackbone(layers.Layer):
    def __init__(self, block, num_blocks):
        super(ResNetBackbone, self).__init__()
        self.in_planes = 64
        self.block = block
        self.num_blocks = num_blocks

        self.conv1 = layers.Conv2D(64, kernel_size=3, strides=1, padding='same', use_bias=False,
                                   kernel_initializer='he_normal',
                                   kernel_regularizer=L2(1e-4)
                                   )
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.ReLU()
        # self.layer1 = self._make_layer(self.block, 32, self.num_blocks[0], stride=1)
        # self.layer2 = self._make_layer(self.block, 48, self.num_blocks[1], stride=2)
        # self.layer3 = self._make_layer(self.block, 64, self.num_blocks[2], stride=2)
        # self.layer4 = self._make_layer(self.block, 128, self.num_blocks[3], stride=1)
        self.layer1 = self._make_layer(self.block, 32, self.num_blocks[0], stride=1)
        self.layer2 = self._make_layer(self.block, 64, self.num_blocks[1], stride=2)
        self.layer3 = self._make_layer(self.block, 128, self.num_blocks[2], stride=2)
        self.layer4 = self._make_layer(self.block, 256, self.num_blocks[3], stride=1)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layer_list = []
        for stride in strides:
            layer_list.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return Sequential([*layer_list])

    def build(self, input_shape):
        self.built = True

    def get_config(self):
        config = {
            'in_planes': self.in_planes,
            'block': self.block,
            'num_blocks': self.num_blocks,
        }
        base_config = super(ResNetBackbone, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs, training=None, mask=None):
        out = self.relu(self.bn1(self.conv1(inputs)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out


def efficient_capsnet_graph(input_shape):
    """
    reimplement for cifar dataset
    """
    inputs = tf.keras.Input(input_shape)
    # (32, 32, 3) ==>> (8, 8, 128)
    x = ResNetBackbone(BasicBlock, [2, 2, 2, 2])(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    # # (4, 4, 128) ==>> (1, 1, 128) ==>> (16, 8)
    # x = PrimaryCaps(128, x.shape[1], 16, 8)(x)
    # (4, 4, 512) ==>> (1, 1, 128) ==>> (16, 8)
    x = PrimaryCaps(256, x.shape[1], 32, 8)(x)

    digit_caps = RoutingA()(x)

    # x = layers.LayerNormalization()(x)
    # digit_caps = FCCaps(10, 16)(x)

    digit_caps_len = Length(name='length_capsnet_output')(digit_caps)

    digit_caps_len = Heterogeneous(num_class=10)((x, digit_caps_len))

    # digit_caps_len = tf.keras.layers.Softmax()(digit_caps_len)

    return tf.keras.Model(inputs=inputs, outputs=[digit_caps, digit_caps_len], name='DWT_Multi_Attention_CapsNet')


def generator_graph(input_shape):
    """
    Generator graph architecture.

    Parameters
    ----------
    input_shape: list
        network input shape
    """
    inputs = tf.keras.Input(16 * 10)

    x = tf.keras.layers.Dense(512, activation='relu', kernel_initializer='he_normal')(inputs)
    x = tf.keras.layers.Dense(1024, activation='relu', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.Dense(np.prod(input_shape), activation='sigmoid', kernel_initializer='glorot_normal')(x)
    x = tf.keras.layers.Reshape(target_shape=input_shape, name='out_generator')(x)

    return tf.keras.Model(inputs=inputs, outputs=x, name='Generator')


def build_graph(input_shape, mode, verbose):
    """
    Efficient-CapsNet graph architecture with reconstruction regularizer.
    The network can be initialize with different modalities.

    Parameters
    ----------
    input_shape: list
        network input shape
    mode: str
        working mode ('train', 'test' & 'play')
    verbose: bool
    """
    inputs = tf.keras.Input(input_shape)
    y_true = tf.keras.layers.Input(shape=(10,))
    noise = tf.keras.layers.Input(shape=(10, 16))

    efficient_capsnet = efficient_capsnet_graph(input_shape)

    if verbose:
        efficient_capsnet.summary()
        print("\n\n")

    digit_caps, digit_caps_len = efficient_capsnet(inputs)
    noised_digitcaps = tf.keras.layers.Add()([digit_caps, noise])  # only if mode is play

    masked_by_y = Mask()([digit_caps, y_true])
    masked = Mask()(digit_caps)
    masked_noised_y = Mask()([noised_digitcaps, y_true])

    generator = generator_graph(input_shape)

    if verbose:
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
