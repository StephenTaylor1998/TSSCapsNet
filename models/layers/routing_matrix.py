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

from tensorflow.keras import layers, models

from .operator_matrix import CapsFPN, CapsFPNTiny


class RoutingBlockMatrix(layers.Layer):
    def __init__(self, routing_name, regularize):
        super(RoutingBlockMatrix, self).__init__()
        if routing_name == "FPN":
            self.fpn = CapsFPN(h=1, d=8)
        elif routing_name == "Tiny_FPN":
            self.fpn = CapsFPNTiny(h=1, d=8)
        else:
            print(f"FPN name {routing_name} should in ['Attention', 'Tiny_FPN', 'FPN']")
            raise NotImplementedError
        self.norm1 = layers.LayerNormalization()
        self.caps_similarity = CapsSimilarity()
        self.norm2 = layers.LayerNormalization()

    def get_config(self):
        return super(RoutingBlockMatrix, self).get_config()

    def call(self, inputs, **kwargs):
        feature_pyramid = self.fpn(inputs)
        feature_pyramid = self.norm1(feature_pyramid)
        caps_similarity = self.caps_similarity(feature_pyramid)
        feature_pyramid = feature_pyramid * caps_similarity
        feature_pyramid = self.norm2(feature_pyramid)
        return feature_pyramid


class RoutingMatrix(layers.Layer):
    """
        Routing for capsule.
    example:
    >>># routing for FPN
    >>>fpn_routing = RoutingMatrix(num_classes=10, routing_name_list=['FPN', 'FPN', 'FPN'], regularize=1e-5)
    >>># routing for FPNTiny
    >>>fpn_tiny_routing = RoutingMatrix(num_classes=10, routing_name_list=['FPNTiny', 'FPNTiny', 'FPNTiny'], regularize=1e-5)
    >>># routing for Attention
    >>>attention_routing = RoutingMatrix(num_classes=10, routing_name_list=['Attention', 'Attention', 'Attention'], regularize=1e-5)
    >>># routing for custom
    >>># custom type
    >>>custom_type = RoutingMatrix(num_classes=10, routing_name_list=['FPN', 'FPNTiny', 'Attention'], regularize=1e-5)
    >>># custom length
    >>>custom_length = RoutingMatrix(num_classes=10,
    >>>                        routing_name_list=['FPN', 'FPN', 'FPN', 'FPN', 'FPN', 'FPN', 'FPN', 'FPN'],
    >>>                        regularize=1e-5)
    """
    def __init__(self, num_classes=10, routing_name_list=None, regularize=1e-5, **kwargs):
        super(RoutingMatrix, self).__init__(**kwargs)
        self.routings = self._make_routing(RoutingBlockMatrix, routing_name_list, regularize)

        # final mapping
        self.final_mapping = CapsuleMapping(num_caps=num_classes, caps_length=16)
        self.norm_final_mapping = layers.LayerNormalization()

    def _make_routing(self, block, routing_name_list: list, regularize):
        layer_list = []
        for routing_name in routing_name_list:
            layer_list.append(block(routing_name, regularize))

        return models.Sequential([*layer_list])

    def get_config(self):
        return super(RoutingMatrix, self).get_config()

    def call(self, inputs, **kwargs):
        feature_pyramid = self.routings(inputs)
        out = self.final_mapping(feature_pyramid)
        out = self.norm_final_mapping(out)
        return out
