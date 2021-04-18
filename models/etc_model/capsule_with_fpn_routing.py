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

import tensorflow as tf
from tensorflow.keras import layers
from models.layers.routing import RoutingA, Routing
from models.layers.layers_efficient import PrimaryCaps, Length
from models.layers.backbone import ResNetBackbone, BasicBlockDWT




# class BasicBlockDWT(layers.Layer):
#     expansion = 1
#
#     def __init__(self, in_planes, planes, stride=1, regularize=1e-4):
#         super(BasicBlockDWT, self).__init__()
#         self.in_planes = in_planes
#         self.planes = planes
#         self.stride = stride
#         if self.stride != 1:
#             self.dwt = DWT()
#         self.conv1 = layers.Conv2D(self.planes, kernel_size=3, strides=1, padding='same', use_bias=False,
#                                    kernel_initializer='he_normal',
#                                    kernel_regularizer=L2(regularize)
#                                    )
#
#         self.bn1 = layers.BatchNormalization()
#         self.conv2 = layers.Conv2D(self.planes, kernel_size=3, strides=1, padding='same', use_bias=False,
#                                    kernel_initializer='he_normal',
#                                    kernel_regularizer=L2(regularize)
#                                    )
#         self.bn2 = layers.BatchNormalization()
#         self.relu = layers.ReLU()
#         self.shortcut = Sequential()
#         if self.stride != 1 or self.in_planes != self.expansion * self.planes:
#             self.shortcut = Sequential([
#                 layers.Conv2D(self.expansion * self.planes, kernel_size=1, strides=self.stride, use_bias=False,
#                               kernel_initializer='he_normal',
#                               kernel_regularizer=L2(regularize)
#                               ),
#                 layers.BatchNormalization()
#             ])
#
#     def get_config(self):
#         return super(BasicBlockDWT, self).get_config()
#
#     def call(self, inputs, **kwargs):
#         if self.stride != 1:
#             out = self.dwt(inputs)
#             out = self.relu(self.bn1(self.conv1(out)))
#         else:
#             out = self.relu(self.bn1(self.conv1(inputs)))
#         out = self.bn2(self.conv2(out))
#         out += self.shortcut(inputs)
#         out = self.relu(out)
#         return out


# class BasicBlock(layers.Layer):
#     expansion = 1
#
#     def __init__(self, in_planes, planes, stride=1, regularize=1e-4):
#         super(BasicBlock, self).__init__()
#         self.in_planes = in_planes
#         self.planes = planes
#         self.stride = stride
#         self.conv1 = layers.Conv2D(self.planes, kernel_size=3, strides=self.stride, padding='same', use_bias=False,
#                                    kernel_initializer='he_normal',
#                                    kernel_regularizer=L2(regularize)
#                                    )
#         self.bn1 = layers.BatchNormalization()
#         self.conv2 = layers.Conv2D(self.planes, kernel_size=3, strides=1, padding='same', use_bias=False,
#                                    kernel_initializer='he_normal',
#                                    kernel_regularizer=L2(regularize)
#                                    )
#         self.bn2 = layers.BatchNormalization()
#         self.relu = layers.ReLU()
#         if self.stride != 1 or self.in_planes != self.expansion * self.planes:
#             self.shortcut = Sequential([
#                 layers.Conv2D(self.expansion * self.planes, kernel_size=1, strides=self.stride, use_bias=False,
#                               kernel_initializer='he_normal',
#                               kernel_regularizer=L2(regularize)
#                               ),
#                 layers.BatchNormalization()
#             ])
#         else:
#             self.shortcut = Sequential()
#
#     def build(self, input_shape):
#         self.built = True
#
#     def get_config(self):
#         return super(BasicBlock, self).get_config()
#
#     def call(self, inputs, **kwargs):
#         out = self.relu(self.bn1(self.conv1(inputs)))
#         out = self.bn2(self.conv2(out))
#         out += self.shortcut(inputs)
#         out = self.relu(out)
#         return out


# class Bottleneck(layers.Layer):
#     expansion = 4
#
#     def __init__(self, in_planes, planes, stride=1, regularize=1e-4):
#         super(Bottleneck, self).__init__()
#         self.in_planes = in_planes
#         self.planes = planes
#         self.stride = stride
#
#         self.conv1 = layers.Conv2D(self.planes, kernel_size=1, use_bias=False,
#                                    kernel_initializer='he_normal',
#                                    kernel_regularizer=L2(regularize)
#                                    )
#         self.bn1 = layers.BatchNormalization()
#         self.conv2 = layers.Conv2D(self.planes, kernel_size=3, strides=self.stride, padding='same', use_bias=False,
#                                    kernel_initializer='he_normal',
#                                    kernel_regularizer=L2(regularize)
#                                    )
#         self.bn2 = layers.BatchNormalization()
#         self.conv3 = layers.Conv2D(self.expansion * self.planes, kernel_size=1, use_bias=False,
#                                    kernel_initializer='he_normal',
#                                    kernel_regularizer=L2(regularize)
#                                    )
#         self.bn3 = layers.BatchNormalization()
#         self.relu = layers.ReLU()
#
#         if self.stride != 1 or self.in_planes != self.expansion * self.planes:
#             self.shortcut = Sequential([
#                 layers.Conv2D(self.expansion * self.planes, kernel_size=1, strides=self.stride, use_bias=False,
#                               kernel_initializer='he_normal',
#                               kernel_regularizer=L2(regularize)
#                               ),
#                 layers.BatchNormalization()
#             ])
#         else:
#             self.shortcut = Sequential()
#
#     def build(self, input_shape):
#         self.built = True
#
#     def get_config(self):
#         config = {
#             'in_planes': self.in_planes,
#             'planes': self.planes,
#             'stride': self.stride,
#         }
#         base_config = super(Bottleneck, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))
#
#     def call(self, inputs, **kwargs):
#         out = self.relu(self.bn1(self.conv1(inputs)))
#         out = self.relu(self.bn2(self.conv2(out)))
#         out = self.bn3(self.conv3(out))
#         out += self.shortcut(inputs)
#         out = self.relu(out)
#         return out


def efficient_capsnet_graph(input_shape, num_classes, routing_name_list=None):
    routing_name_list = ['FPN', 'FPN', 'FPN'] if routing_name_list is None else routing_name_list
    inputs = tf.keras.Input(input_shape)
    # (32, 32, 3) ==>> (8, 8, 128)
    x = ResNetBackbone(BasicBlockDWT, [2, 2, 2, 2])(inputs)
    # x = ResNetBackbone(BasicBlock, [2, 2, 2, 2])(inputs)
    # x = layers.Conv2D(256, 1)(inputs)
    x = layers.BatchNormalization()(x)
    # (4, 4, 256) ==>> (1, 1, 256) ==>> (32, 8)
    x = PrimaryCaps(256, x.shape[1], 32, 8)(x)
    # # (4, 4, 512) ==>> (1, 1, 512) ==>> (64, 8)
    # x = PrimaryCaps(512, x.shape[1], 64, 8)(x)
    # digit_caps = RoutingA(num_classes)(x)
    digit_caps = Routing(num_classes, routing_name_list, regularize=1e-5)(x)
    # x = layers.LayerNormalization()(x)
    digit_caps_len = Length()(digit_caps)
    # digit_caps_len = Heterogeneous(num_class=10)((x, digit_caps_len))
    return tf.keras.Model(inputs=[inputs], outputs=[digit_caps_len])


def build_graph(input_shape, num_classes, routing_name_list):
    """
    Efficient-CapsNet graph architecture with reconstruction regularizer.
    The network can be initialize with different modalities.
    Parameters
    ----------
    input_shape: list
        network input shape
        :param input_shape: model input shape
        :param num_classes: number of classes
        :param routing_name_list: name list for routing name
            for example routing_name_list = ['FPN', 'FPN', 'FPN']
    """
    efficient_capsnet = efficient_capsnet_graph(input_shape, num_classes, routing_name_list)
    efficient_capsnet.summary()
    return efficient_capsnet
