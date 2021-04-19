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

from tensorflow.keras import Model, layers, Sequential
from tensorflow.keras.regularizers import L2

from models.layers.transform.dwt import DWT


class BasicBlock(layers.Layer):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, regularize=1e-4):
        super(BasicBlock, self).__init__()
        self.in_planes = in_planes
        self.planes = planes
        self.stride = stride
        self.conv1 = layers.Conv2D(self.planes, kernel_size=3, strides=self.stride, padding='same', use_bias=False,
                                   kernel_initializer='he_normal',
                                   kernel_regularizer=L2(regularize)
                                   )
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(self.planes, kernel_size=3, strides=1, padding='same', use_bias=False,
                                   kernel_initializer='he_normal',
                                   kernel_regularizer=L2(regularize)
                                   )
        self.bn2 = layers.BatchNormalization()
        self.relu = layers.ReLU()
        self.shortcut = Sequential()
        if self.stride != 1 or self.in_planes != self.expansion * self.planes:
            self.shortcut = Sequential([
                layers.Conv2D(self.expansion * self.planes, kernel_size=1, strides=self.stride, use_bias=False,
                              kernel_initializer='he_normal',
                              kernel_regularizer=L2(regularize)
                              ),
                layers.BatchNormalization()
            ])

    def build(self, input_shape):
        self.built = True

    def get_config(self):
        return super(BasicBlock, self).get_config()

    def call(self, inputs, **kwargs):
        out = self.relu(self.bn1(self.conv1(inputs)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(inputs)
        out = self.relu(out)
        return out


class Bottleneck(layers.Layer):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, regularize=1e-4):
        super(Bottleneck, self).__init__()
        self.in_planes = in_planes
        self.planes = planes
        self.stride = stride

        self.conv1 = layers.Conv2D(self.planes, kernel_size=1, use_bias=False,
                                   kernel_initializer='he_normal',
                                   kernel_regularizer=L2(regularize)
                                   )
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(self.planes, kernel_size=3, strides=self.stride, padding='same', use_bias=False,
                                   kernel_initializer='he_normal',
                                   kernel_regularizer=L2(regularize)
                                   )
        self.bn2 = layers.BatchNormalization()
        self.conv3 = layers.Conv2D(self.expansion * self.planes, kernel_size=1, use_bias=False,
                                   kernel_initializer='he_normal',
                                   kernel_regularizer=L2(regularize)
                                   )
        self.bn3 = layers.BatchNormalization()
        self.relu = layers.ReLU()
        self.shortcut = Sequential()
        if self.stride != 1 or self.in_planes != self.expansion * self.planes:
            self.shortcut = Sequential([
                layers.Conv2D(self.expansion * self.planes, kernel_size=1, strides=self.stride, use_bias=False,
                              kernel_initializer='he_normal',
                              kernel_regularizer=L2(regularize)
                              ),
                layers.BatchNormalization()
            ])

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


class TinyBlockDWT(layers.Layer):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, regularize=1e-4):
        super(TinyBlockDWT, self).__init__()
        self.in_planes = in_planes
        self.planes = planes
        self.stride = stride
        if self.stride == 2:
            self.conv1 = DWT()
        else:
            self.conv1 = layers.Conv2D(self.planes/2, kernel_size=3, strides=1, padding='same', use_bias=False,
                                       kernel_initializer='he_normal',
                                       kernel_regularizer=L2(regularize)
                                       )

        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(self.planes, kernel_size=1, strides=1, padding='same', use_bias=False,
                                   kernel_initializer='he_normal',
                                   kernel_regularizer=L2(regularize)
                                   )
        self.bn2 = layers.BatchNormalization()
        self.relu = layers.ReLU()
        self.shortcut = Sequential()
        if self.stride != 1 or self.in_planes != self.expansion * self.planes:
            self.shortcut = Sequential([
                layers.Conv2D(self.expansion * self.planes, kernel_size=1, strides=self.stride, use_bias=False,
                              kernel_initializer='he_normal',
                              kernel_regularizer=L2(regularize)
                              ),
                layers.BatchNormalization()
            ])

    def get_config(self):
        return super(TinyBlockDWT, self).get_config()

    def call(self, inputs, **kwargs):
        out = self.relu(self.bn1(self.conv1(inputs)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(inputs)
        out = self.relu(out)
        return out


class TinyBottleDWT(layers.Layer):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, regularize=1e-4):
        super(TinyBottleDWT, self).__init__()
        self.in_planes = in_planes
        self.planes = planes
        self.stride = stride

        self.conv1 = layers.Conv2D(self.planes, kernel_size=1, use_bias=False,
                                   kernel_initializer='he_normal',
                                   kernel_regularizer=L2(regularize)
                                   )
        self.bn1 = layers.BatchNormalization()
        if stride == 2:
            self.conv2 = DWT()
        else:
            self.conv2 = layers.Conv2D(self.planes/2, kernel_size=3, strides=self.stride,
                                       padding='same', use_bias=False,
                                       kernel_initializer='he_normal',
                                       kernel_regularizer=L2(regularize)
                                       )
        self.bn2 = layers.BatchNormalization()
        self.conv3 = layers.Conv2D(self.expansion * self.planes, kernel_size=1, use_bias=False,
                                   kernel_initializer='he_normal',
                                   kernel_regularizer=L2(regularize)
                                   )
        self.bn3 = layers.BatchNormalization()
        self.relu = layers.ReLU()
        self.shortcut = Sequential()
        if self.stride != 1 or self.in_planes != self.expansion * self.planes:
            self.shortcut = Sequential([
                layers.Conv2D(self.expansion * self.planes, kernel_size=1, strides=self.stride, use_bias=False,
                              kernel_initializer='he_normal',padding='same',
                              kernel_regularizer=L2(regularize)
                              ),
                layers.BatchNormalization()
            ])

    def build(self, input_shape):
        self.built = True

    def get_config(self):
        config = {
            'in_planes': self.in_planes,
            'planes': self.planes,
            'stride': self.stride,
        }
        base_config = super(TinyBottleDWT, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs, **kwargs):
        out = self.relu(self.bn1(self.conv1(inputs)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(inputs)
        out = self.relu(out)
        return out


class BasicBlockDWT(layers.Layer):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, regularize=1e-4):
        super(BasicBlockDWT, self).__init__()
        self.in_planes = in_planes
        self.planes = planes
        self.stride = stride
        if self.stride == 2:
            self.conv1 = DWT()
        else:
            self.conv1 = layers.Conv2D(self.planes, kernel_size=3, strides=1, padding='same', use_bias=False,
                                       kernel_initializer='he_normal',
                                       kernel_regularizer=L2(regularize)
                                       )

        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(self.planes, kernel_size=3, strides=1, padding='same', use_bias=False,
                                   kernel_initializer='he_normal',
                                   kernel_regularizer=L2(regularize)
                                   )
        self.bn2 = layers.BatchNormalization()
        self.relu = layers.ReLU()
        self.shortcut = Sequential()
        if self.stride != 1 or self.in_planes != self.expansion * self.planes:
            self.shortcut = Sequential([
                layers.Conv2D(self.expansion * self.planes, kernel_size=1, strides=self.stride, use_bias=False,
                              kernel_initializer='he_normal',
                              kernel_regularizer=L2(regularize)
                              ),
                layers.BatchNormalization()
            ])

    def get_config(self):
        return super(BasicBlockDWT, self).get_config()

    def call(self, inputs, **kwargs):
        out = self.relu(self.bn1(self.conv1(inputs)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(inputs)
        out = self.relu(out)
        return out


class BottleneckDWT(layers.Layer):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, regularize=1e-4):
        super(BottleneckDWT, self).__init__()
        self.in_planes = in_planes
        self.planes = planes
        self.stride = stride

        self.conv1 = layers.Conv2D(self.planes, kernel_size=1, use_bias=False,
                                   kernel_initializer='he_normal',
                                   kernel_regularizer=L2(regularize)
                                   )
        self.bn1 = layers.BatchNormalization()
        if stride == 2:
            self.conv2 = DWT()
        else:
            self.conv2 = layers.Conv2D(self.planes, kernel_size=3, strides=self.stride, padding='same', use_bias=False,
                                       kernel_initializer='he_normal',
                                       kernel_regularizer=L2(regularize)
                                       )
        self.bn2 = layers.BatchNormalization()
        self.conv3 = layers.Conv2D(self.expansion * self.planes, kernel_size=1, use_bias=False,
                                   kernel_initializer='he_normal',
                                   kernel_regularizer=L2(regularize)
                                   )
        self.bn3 = layers.BatchNormalization()
        self.relu = layers.ReLU()
        self.shortcut = Sequential()
        if self.stride != 1 or self.in_planes != self.expansion * self.planes:
            self.shortcut = Sequential([
                layers.Conv2D(self.expansion * self.planes, kernel_size=1, strides=self.stride, use_bias=False,
                              kernel_initializer='he_normal',
                              kernel_regularizer=L2(regularize)
                              ),
                layers.BatchNormalization()
            ])

    def build(self, input_shape):
        self.built = True

    def get_config(self):
        config = {
            'in_planes': self.in_planes,
            'planes': self.planes,
            'stride': self.stride,
        }
        base_config = super(BottleneckDWT, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs, **kwargs):
        out = self.relu(self.bn1(self.conv1(inputs)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(inputs)
        out = self.relu(out)
        return out


class ResNet(Model):
    def __init__(self, block, num_blocks, num_classes=10, half=True, regularize=1e-4):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = layers.Conv2D(64, kernel_size=3, strides=1, padding='same', use_bias=False,
                                   kernel_initializer='he_normal',
                                   kernel_regularizer=L2(regularize)
                                   )
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.ReLU()
        if half:
            self.layer1 = self._make_layer(block, 32, num_blocks[0], stride=1, regularize=regularize)
            self.layer2 = self._make_layer(block, 64, num_blocks[1], stride=2, regularize=regularize)
            self.layer3 = self._make_layer(block, 128, num_blocks[2], stride=2, regularize=regularize)
            self.layer4 = self._make_layer(block, 256, num_blocks[3], stride=2, regularize=regularize)
        else:
            self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, regularize=regularize)
            self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, regularize=regularize)
            self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, regularize=regularize)
            self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, regularize=regularize)

        self.pool = layers.GlobalAveragePooling2D()
        self.linear = layers.Dense(num_classes, activation='softmax')

    def _make_layer(self, block, planes, num_blocks, stride, regularize):
        strides = [stride] + [1] * (num_blocks - 1)
        layer_list = []
        for stride in strides:
            layer_list.append(block(self.in_planes, planes, stride, regularize))
            self.in_planes = planes * block.expansion
        return Sequential([*layer_list])

    def build(self, input_shape):
        self.built = True

    def get_config(self):
        config = {
            'in_planes': self.in_planes,
            'block': self.block,
        }
        base_config = super(ResNet, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs, training=None, mask=None):
        out = self.relu(self.bn1(self.conv1(inputs)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.pool(out)
        out = self.linear(out)
        return out


class ResNetBackbone(layers.Layer):
    def __init__(self, block, num_blocks, half=True, regularize=1e-4):
        super(ResNetBackbone, self).__init__()
        self.in_planes = 64
        self.num_blocks = num_blocks
        self.conv1 = layers.Conv2D(64, kernel_size=3, strides=1, padding='same', use_bias=False,
                                   kernel_initializer='he_normal',
                                   kernel_regularizer=L2(regularize)
                                   )
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.ReLU()
        if half:
            self.layer1 = self._make_layer(block, 32, self.num_blocks[0], stride=1, regularize=regularize)
            self.layer2 = self._make_layer(block, 64, self.num_blocks[1], stride=2, regularize=regularize)
            self.layer3 = self._make_layer(block, 128, self.num_blocks[2], stride=2, regularize=regularize)
            self.layer4 = self._make_layer(block, 256, self.num_blocks[3], stride=2, regularize=regularize)
        else:
            self.layer1 = self._make_layer(block, 64, self.num_blocks[0], stride=1, regularize=regularize)
            self.layer2 = self._make_layer(block, 128, self.num_blocks[1], stride=2, regularize=regularize)
            self.layer3 = self._make_layer(block, 256, self.num_blocks[2], stride=2, regularize=regularize)
            self.layer4 = self._make_layer(block, 512, self.num_blocks[3], stride=2, regularize=regularize)

    def _make_layer(self, block, planes, num_blocks, stride, regularize):
        strides = [stride] + [1] * (num_blocks - 1)
        layer_list = []
        for stride in strides:
            layer_list.append(block(self.in_planes, planes, stride, regularize))
            self.in_planes = planes * block.expansion
        return Sequential([*layer_list])

    def build(self, input_shape):
        self.built = True

    def get_config(self):
        return super(ResNetBackbone, self).get_config()

    def call(self, inputs, training=None, mask=None):
        out = self.relu(self.bn1(self.conv1(inputs)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out


def resnet18_cifar(block=BasicBlock, num_blocks=None, num_classes=10, half=True, backbone=False):
    if num_blocks is None:
        num_blocks = [2, 2, 2, 2]

    if backbone:
        return ResNetBackbone(block, num_blocks, half=half)
    else:
        return ResNet(block, num_blocks, num_classes=num_classes, half=half)


def resnet34_cifar(block=BasicBlock, num_blocks=None, num_classes=10, half=True, backbone=False):
    if num_blocks is None:
        num_blocks = [3, 4, 6, 3]
    if backbone:
        return ResNetBackbone(block, num_blocks, half=half)
    else:
        return ResNet(block, num_blocks, num_classes=num_classes, half=half)


def resnet50_cifar(block=Bottleneck, num_blocks=None, num_classes=10, half=True, backbone=False):
    if num_blocks is None:
        num_blocks = [3, 4, 6, 3]
    if backbone:
        return ResNetBackbone(block, num_blocks, half=half)
    else:
        return ResNet(block, num_blocks, num_classes=num_classes, half=half)


def resnet101_cifar(block=Bottleneck, num_blocks=None, num_classes=10, half=True, backbone=False):
    if num_blocks is None:
        num_blocks = [3, 4, 23, 3]
    if backbone:
        return ResNetBackbone(block, num_blocks, half=half)
    else:
        return ResNet(block, num_blocks, num_classes=num_classes, half=half)


def resnet152_cifar(block=Bottleneck, num_blocks=None, num_classes=10, half=True, backbone=False):
    if num_blocks is None:
        num_blocks = [3, 8, 36, 3]
    if backbone:
        return ResNetBackbone(block, num_blocks, half=half)
    else:
        return ResNet(block, num_blocks, num_classes=num_classes, half=half)