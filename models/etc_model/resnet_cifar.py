from tensorflow.keras.regularizers import L2
from tensorflow.keras import Input, Model, layers, Sequential


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
        self.shortcut = Sequential()
        if self.stride != 1 or self.in_planes != self.expansion * self.planes:
            self.shortcut = Sequential([
                layers.Conv2D(self.expansion * self.planes, kernel_size=1, strides=self.stride, use_bias=False,
                              kernel_initializer='he_normal',
                              kernel_regularizer=L2(1e-4)
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

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.in_planes = in_planes
        self.planes = planes
        self.stride = stride

        self.conv1 = layers.Conv2D(self.planes, kernel_size=1, use_bias=False,
                                   kernel_initializer='he_normal',
                                   kernel_regularizer=L2(1e-4)
                                   )
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(self.planes, kernel_size=3, strides=self.stride, padding='same', use_bias=False,
                                   kernel_initializer='he_normal',
                                   kernel_regularizer=L2(1e-4)
                                   )
        self.bn2 = layers.BatchNormalization()
        self.conv3 = layers.Conv2D(self.expansion * self.planes, kernel_size=1, use_bias=False,
                                   kernel_initializer='he_normal',
                                   kernel_regularizer=L2(1e-4)
                                   )
        self.bn3 = layers.BatchNormalization()
        self.relu = layers.ReLU()
        self.shortcut = Sequential()
        if self.stride != 1 or self.in_planes != self.expansion * self.planes:
            self.shortcut = Sequential([
                layers.Conv2D(self.expansion * self.planes, kernel_size=1, strides=self.stride, use_bias=False,
                              kernel_initializer='he_normal',
                              kernel_regularizer=L2(1e-4)
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


class ResNet(Model):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__(name="RESNET")
        self.in_planes = 64
        self.block = block
        self.num_blocks = num_blocks
        self.num_classes = num_classes
        self.conv1 = layers.Conv2D(64, kernel_size=3, strides=1, padding='same', use_bias=False,
                                   kernel_initializer='he_normal',
                                   kernel_regularizer=L2(1e-4)
                                   )
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.ReLU()
        # self.layer1 = self._make_layer(self.block, 64, self.num_blocks[0], stride=1)
        # self.layer2 = self._make_layer(self.block, 128, self.num_blocks[1], stride=2)
        # self.layer3 = self._make_layer(self.block, 256, self.num_blocks[2], stride=2)
        # self.layer4 = self._make_layer(self.block, 512, self.num_blocks[3], stride=2)
        self.layer1 = self._make_layer(self.block, 32, self.num_blocks[0], stride=1)
        self.layer2 = self._make_layer(self.block, 64, self.num_blocks[1], stride=2)
        self.layer3 = self._make_layer(self.block, 128, self.num_blocks[2], stride=2)
        self.layer4 = self._make_layer(self.block, 256, self.num_blocks[3], stride=2)
        self.pool = layers.GlobalAveragePooling2D()
        self.linear = layers.Dense(self.num_classes, activation='softmax')

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
        return super(ResNet, self).get_config()

    def call(self, inputs, training=None, mask=None):
        out = self.relu(self.bn1(self.conv1(inputs)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.pool(out)
        out = self.linear(out)
        return out


def resnet18_cifar(block=BasicBlock, num_blocks=None, num_classes=10):
    if num_blocks is None:
        num_blocks = [2, 2, 2, 2]
    return ResNet(block, num_blocks, num_classes=num_classes)


def resnet34_cifar(block=BasicBlock, num_blocks=None, num_classes=10):
    if num_blocks is None:
        num_blocks = [3, 4, 6, 3]
    return ResNet(block, num_blocks, num_classes=num_classes)


def resnet50_cifar(block=Bottleneck, num_blocks=None, num_classes=10):
    if num_blocks is None:
        num_blocks = [3, 4, 6, 3]
    return ResNet(block, num_blocks, num_classes=num_classes)


def resnet101_cifar(block=Bottleneck, num_blocks=None, num_classes=10):
    if num_blocks is None:
        num_blocks = [3, 4, 23, 3]
    return ResNet(block, num_blocks, num_classes=num_classes)


def resnet152_cifar(block=Bottleneck, num_blocks=None, num_classes=10):
    if num_blocks is None:
        num_blocks = [3, 8, 36, 3]
    return ResNet(block, num_blocks, num_classes=num_classes)


def build_graph(input_shape, num_classes=10, depth=18):
    if depth == 18:
        model = resnet18_cifar(num_classes=num_classes)

    elif depth == 34:
        model = resnet34_cifar(num_classes=num_classes)

    elif depth == 50:
        model = resnet50_cifar(num_classes=num_classes)

    elif depth == 101:
        model = resnet101_cifar(num_classes=num_classes)

    elif depth == 152:
        model = resnet152_cifar(num_classes=num_classes)

    else:
        raise NotImplemented

    print(model.name)
    input_layer = Input(input_shape)
    out = model(input_layer)
    build_model = Model(inputs=[input_layer], outputs=[out])
    # return build_model
    return model


# if __name__ == '__main__':
#     import tensorflow as tf
#     import numpy as np
#
#     mnist_model = build_graph(input_shape=(28, 28, 1), depth=152)
#     mnist_model.compile(
#         optimizer="adam",
#         loss=tf.keras.losses.SparseCategoricalCrossentropy(),
#         metrics=['sparse_categorical_accuracy']
#     )
#     (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
#     x_train, x_test = np.expand_dims(x_train, -1) / 255.0, np.expand_dims(x_test, -1) / 255.0
#     print(x_train.shape)
#     mnist_model.summary()
#     mnist_model.fit(
#         x_test,
#         y_test,
#         batch_size=32,
#         epochs=1,
#         validation_data=(x_train, y_train),
#         validation_freq=1
#     )
#     mnist_model.save_weights('./resnet18_mnist.h5')
