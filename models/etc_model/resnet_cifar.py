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

from tensorflow.keras import Input, Model
from models.layers.backbone import resnet18_cifar, resnet34_cifar, resnet50_cifar, \
    resnet101_cifar, resnet152_cifar


def build_graph(input_shape, num_classes=10, depth=18, half=True):
    if depth == 18:
        model = resnet18_cifar(num_classes=num_classes, half=half)

    elif depth == 34:
        model = resnet34_cifar(num_classes=num_classes, half=half)

    elif depth == 50:
        model = resnet50_cifar(num_classes=num_classes, half=half)

    elif depth == 101:
        model = resnet101_cifar(num_classes=num_classes, half=half)

    elif depth == 152:
        model = resnet152_cifar(num_classes=num_classes, half=half)

    else:
        raise NotImplemented

    print(model.name)
    input_layer = Input(input_shape)
    out = model(input_layer)
    build_model = Model(inputs=[input_layer], outputs=[out])
    build_model.summary()
    print("\n")
    del build_model
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
