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
from .layers_hinton import squash
from .attention import CapsuleAttentionBlock, BaselineAttention
from tensorflow.keras import layers, models
from .operators import CapsFPN, Q, CapsSimilarity, CapsuleMappingTiny, CapsuleMapping, CapsFPNTiny


class AttentionDigitCaps(tf.keras.layers.Layer):

    def __init__(self, C, L, routing=None, kernel_initializer='glorot_uniform', **kwargs):
        super(AttentionDigitCaps, self).__init__(**kwargs)
        self.C = C
        self.L = L
        self.routing = routing
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.attention = None

    def build(self, input_shape):
        assert len(input_shape) >= 5, "The input Tensor should have shape=[None,H,W,input_C,input_L]"
        H = input_shape[-4]
        W = input_shape[-3]
        input_C = input_shape[-2]
        input_L = input_shape[-1]

        self.W = self.add_weight(shape=[H * W * input_C, input_L, self.L * self.C], initializer=self.kernel_initializer,
                                 name='W')
        self.biases = self.add_weight(shape=[self.C, self.L], initializer='zeros', name='biases')
        self.attention = CapsuleAttentionBlock()
        self.built = True

    def call(self, inputs, **kwargs):
        H, W, input_C, input_L = inputs.shape[1:]  # input shape=(None,H,W,input_C,input_L)
        x = tf.reshape(inputs, (-1, H * W * input_C, input_L))  # x shape=(None,H*W*input_C,input_L)

        u = tf.einsum('...ji,jik->...jk', x, self.W)  # u shape=(None,H*W*input_C,C*L)
        u = tf.reshape(u, (-1, H * W * input_C, self.C, self.L))  # u shape=(None,H*W*input_C,C,L)

        if self.routing == 'attention':
            s = tf.reduce_sum(u, axis=1, keepdims=True)
            s += self.biases
            s = self.attention(s[:, 0, ...])
            s = tf.expand_dims(s, 1)
            v = squash(s)
            v = v[:, 0, ...]
        elif self.routing:
            # Hinton's routing
            b = tf.zeros(tf.shape(u)[:-1])[..., None]  # b shape=(None,H*W*input_C,C,1) -> (None,i,j,1)
            for r in range(self.routing):
                c = tf.nn.softmax(b, axis=2)  # c shape=(None,H*W*input_C,C,1) -> (None,i,j,1)
                s = tf.reduce_sum(tf.multiply(u, c), axis=1, keepdims=True)  # s shape=(None,1,C,L)
                s += self.biases
                s = self.attention(s[:, 0, ...])
                s = tf.expand_dims(s, 1)
                v = squash(s)  # v shape=(None,1,C,L)
                if r < self.routing - 1:
                    b += tf.reduce_sum(tf.multiply(u, v), axis=-1, keepdims=True)
            v = v[:, 0, ...]  # v shape=(None,C,L)
        else:
            s = tf.reduce_sum(u, axis=1, keepdims=True)
            s += self.biases
            v = squash(s)
            v = v[:, 0, ...]
        return v

    def compute_output_shape(self, input_shape):
        return None, self.C, self.L

    def get_config(self):
        config = {
            'C': self.C,
            'L': self.L,
            'routing': self.routing
        }
        base_config = super(AttentionDigitCaps, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class RoutingTiny(layers.Layer):
    def __init__(self, **kwargs):
        super(RoutingTiny, self).__init__(**kwargs)
        self.fpn = CapsFPNTiny(out_length=8)
        self.q = Q()
        self.k = CapsSimilarity()
        self.mapping = CapsuleMappingTiny()

    def call(self, inputs, **kwargs):
        feature_pyramid = self.fpn(inputs)
        q = self.q(feature_pyramid)
        k = self.k((feature_pyramid))
        out = self.mapping((k, q, feature_pyramid))

        return out


class RoutingA(layers.Layer):
    def __init__(self, num_classes=10, **kwargs):
        super(RoutingA, self).__init__(**kwargs)
        # self.fpn1 = BaselineAttention(h=1, d=8)
        # self.fpn1 = CapsFPNTiny(out_length=8)
        self.fpn1 = CapsFPN(num_caps=[16, 8, 4, 4], length=8)
        self.norm1 = layers.LayerNormalization()
        self.caps_similarity1 = CapsSimilarity()
        self.norm_caps_similarity1 = layers.LayerNormalization()

        # self.fpn2 = BaselineAttention(h=1, d=8)
        # self.fpn2 = CapsFPNTiny(out_length=8)
        self.fpn2 = CapsFPN(num_caps=[16, 8, 4, 4], length=8)
        self.norm2 = layers.LayerNormalization()
        self.caps_similarity2 = CapsSimilarity()
        self.norm_caps_similarity2 = layers.LayerNormalization()

        # self.fpn3 = BaselineAttention(h=1, d=8)
        # self.fpn3 = CapsFPNTiny(out_length=8)
        self.fpn3 = CapsFPN(num_caps=[16, 8, 4, 4], length=8)
        self.norm3 = layers.LayerNormalization()
        self.caps_similarity3 = CapsSimilarity()
        self.norm_caps_similarity3 = layers.LayerNormalization()
        # final mapping
        self.final_mapping = CapsuleMapping(num_caps=num_classes, caps_length=16)
        self.norm_final_mapping = layers.LayerNormalization()

    def get_config(self):
        return super(RoutingA, self).get_config()

    def call(self, inputs, **kwargs):
        feature_pyramid = self.fpn1(inputs)
        feature_pyramid = self.norm1(feature_pyramid)
        caps_similarity = self.caps_similarity1(feature_pyramid)
        feature_pyramid = feature_pyramid * caps_similarity
        feature_pyramid = self.norm_caps_similarity1(feature_pyramid)

        feature_pyramid = self.fpn2(feature_pyramid)
        feature_pyramid = self.norm2(feature_pyramid)
        caps_similarity = self.caps_similarity2(feature_pyramid)
        feature_pyramid = feature_pyramid * caps_similarity
        feature_pyramid = self.norm_caps_similarity2(feature_pyramid)

        feature_pyramid = self.fpn3(feature_pyramid)
        feature_pyramid = self.norm3(feature_pyramid)
        caps_similarity = self.caps_similarity3(feature_pyramid)
        feature_pyramid = feature_pyramid * caps_similarity
        feature_pyramid = self.norm_caps_similarity3(feature_pyramid)

        out = self.final_mapping(feature_pyramid)
        out = self.norm_final_mapping(out)
        return out


class RoutingBlock(layers.Layer):
    def __init__(self, routing_name, regularize):
        super(RoutingBlock, self).__init__()
        if routing_name == "Attention":
            self.fpn = BaselineAttention(h=1, d=8)
        elif routing_name == "Tiny_FPN":
            self.fpn = CapsFPNTiny(out_length=8, regularize=regularize)
        elif routing_name == "FPN":
            self.fpn = CapsFPN(num_caps=[16, 8, 4, 4], length=8)
        else:
            print(f"FPN name {routing_name} should in ['Attention', 'Tiny_FPN', 'FPN']")
            raise NotImplementedError
        self.norm1 = layers.LayerNormalization()
        self.caps_similarity = CapsSimilarity()
        self.norm2 = layers.LayerNormalization()

    def get_config(self):
        return super(RoutingBlock, self).get_config()

    def call(self, inputs, **kwargs):
        feature_pyramid = self.fpn(inputs)
        feature_pyramid = self.norm1(feature_pyramid)
        caps_similarity = self.caps_similarity(feature_pyramid)
        feature_pyramid = feature_pyramid * caps_similarity
        feature_pyramid = self.norm2(feature_pyramid)
        return feature_pyramid


class Routing(layers.Layer):
    """
        Routing for capsule.
    example:
    >>># routing for FPN
    >>>fpn_routing = Routing(num_classes=10, routing_name_list=['FPN', 'FPN', 'FPN'], regularize=1e-5)
    >>># routing for FPNTiny
    >>>fpn_tiny_routing = Routing(num_classes=10, routing_name_list=['FPNTiny', 'FPNTiny', 'FPNTiny'], regularize=1e-5)
    >>># routing for Attention
    >>>attention_routing = Routing(num_classes=10, routing_name_list=['Attention', 'Attention', 'Attention'], regularize=1e-5)
    >>># routing for custom
    >>># custom type
    >>>custom_type = Routing(num_classes=10, routing_name_list=['FPN', 'FPNTiny', 'Attention'], regularize=1e-5)
    >>># custom length
    >>>custom_length = Routing(num_classes=10,
    >>>                        routing_name_list=['FPN', 'FPN', 'FPN', 'FPN', 'FPN', 'FPN', 'FPN', 'FPN'],
    >>>                        regularize=1e-5)
    """
    def __init__(self, num_classes=10, routing_name_list=None, regularize=1e-5, **kwargs):
        super(Routing, self).__init__(**kwargs)
        self.routings = self._make_routing(RoutingBlock, routing_name_list, regularize)

        # final mapping
        self.final_mapping = CapsuleMapping(num_caps=num_classes, caps_length=16)
        self.norm_final_mapping = layers.LayerNormalization()

    def _make_routing(self, block, routing_name_list: list, regularize):
        layer_list = []
        for routing_name in routing_name_list:
            layer_list.append(block(routing_name, regularize))

        return models.Sequential([*layer_list])

    def get_config(self):
        return super(Routing, self).get_config()

    def call(self, inputs, **kwargs):
        feature_pyramid = self.routings(inputs)
        out = self.final_mapping(feature_pyramid)
        out = self.norm_final_mapping(out)
        return out
