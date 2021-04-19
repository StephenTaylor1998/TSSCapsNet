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
from .gumbel import GumbelSoftmax


class GumbelGate(tf.keras.layers.Layer):

    def __init__(self, act='relu'):
        super(GumbelGate, self).__init__()
        self.batch_size = None
        self.in_channel = None
        self.avg_pool = layers.GlobalAveragePooling2D()
        self.reshape = None
        self.inp_gate = None
        self.inp_gate_l = None
        self.channel_reshape = None
        self.gumbel_softmax = GumbelSoftmax()

        if act == 'relu':
            self.relu = layers.ReLU()
        elif act == 'relu6':
            self.relu = layers.ReLU(max_value=6.)
        else:
            self.relu = None
            raise NotImplementedError

    def build(self, input_shape):
        # self.batch_size = input_shape[-4]
        self.in_channel = input_shape[-1]
        self.reshape = layers.Reshape((1, 1, self.in_channel))
        self.inp_gate = tf.keras.Sequential([
            layers.Conv2D(self.in_channel, kernel_size=1, strides=1, use_bias=True),
            layers.BatchNormalization(),
            self.relu,
        ])
        self.inp_gate_l = layers.Conv2D(self.in_channel * 2, kernel_size=1, strides=1, groups=self.in_channel,
                                        use_bias=True)
        self.channel_reshape = layers.Reshape((self.in_channel, 2))
        self.built = True
        
    def get_config(self):
        return super(GumbelGate, self).get_config()

    def call(self, inputs, temperature=1., **kwargs):
        # (batch, h, w, channel) ==>> (batch, channel)
        hatten = self.avg_pool(inputs)
        # (batch, channel) ==>> (batch, 1, 1, channel)
        hatten = self.reshape(hatten)
        # (batch, 1, 1, channel) ==>> (batch, 1, 1, channel)
        hatten_d = self.inp_gate(hatten)
        # (batch, 1, 1, channel) ==>> (batch, 1, 1, channel*2)
        hatten_d = self.inp_gate_l(hatten_d)
        # (batch, 1, 1, channel) ==>> (batch, channel, 2)
        hatten_d = self.channel_reshape(hatten_d)
        # (batch, channel, 2) ==>> (batch, channel)
        hatten_d = self.gumbel_softmax(hatten_d, temp=temperature, force_hard=True)
        # hatten_d = self.gumbel_softmax(hatten_d, temp=temperature, force_hard=False)
        # (batch, channel) ==>> (batch, 1, 1, channel)
        hatten_d = self.reshape(hatten_d)
        # (batch, h, w, channel) * (batch, 1, 1, channel) ==>> (batch, h, w, channel)
        x = inputs * hatten_d
        return x, hatten_d


class DynamicGate(tf.keras.layers.Layer):

    def __init__(self, act='relu'):
        super(DynamicGate, self).__init__()
        self.in_channel = None
        self.avg_pool = layers.GlobalAveragePooling2D()
        self.max_pool = layers.GlobalMaxPooling2D()
        self.factor_for_pools = self.add_weight(
            name="factor_for_pools", shape=[2], trainable=True,
            initializer=tf.initializers.Ones(), )
        self.reshape = None
        self.inp_gate = None
        self.inp_gate_l = None
        self.channel_reshape = None
        self.gumbel_softmax = GumbelSoftmax()

        if act == 'relu':
            self.relu = layers.ReLU()
        elif act == 'relu6':
            self.relu = layers.ReLU(max_value=6.)
        else:
            self.relu = None
            raise NotImplementedError

    def build(self, input_shape):
        self.in_channel = input_shape[-1]
        self.reshape = layers.Reshape((1, 1, self.in_channel))
        self.inp_gate = tf.keras.Sequential([
            layers.Conv2D(self.in_channel, kernel_size=1, strides=1, use_bias=True),
            layers.BatchNormalization(),
            self.relu,
        ])
        self.inp_gate_l = layers.Conv2D(self.in_channel * 2, kernel_size=1, strides=1, groups=self.in_channel,
                                        use_bias=True)
        self.channel_reshape = layers.Reshape((self.in_channel, 2))
        self.built = True
        
    def get_config(self):
        super(DynamicGate, self).get_config()

    def call(self, inputs, temperature=1., **kwargs):
        avg_pool = self.avg_pool(inputs)
        max_pool = self.max_pool(inputs)
        factors = tf.math.softmax(self.factor_for_pools)
        # pool = self.reshape(max_pool + avg_pool)
        print(factors)
        pool = self.reshape(avg_pool * factors[0] + max_pool * factors[1])
        l1 = self.inp_gate(pool)
        l2 = self.inp_gate_l(l1)
        hatten_d = self.channel_reshape(l2)
        sample = self.gumbel_softmax(hatten_d, temp=temperature, force_hard=True)
        sample = self.reshape(sample)
        x = inputs * sample
        return x, sample
