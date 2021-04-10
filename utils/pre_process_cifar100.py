# Copyright 2021 Adam Byerly & Vittorio Mazzia & Francesco Salvetti. All Rights Reserved.
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
from utils.pre_process_cifar10 import image_rotate_random, image_shift_rand, image_squish_random, generator, \
    image_erase_random


tf2 = tf.compat.v2

# constants
CIFAR_IMG_SIZE = 32
CIFAR_IMG_CHANNEL = 3
CIFAR_TRAIN_IMAGE_COUNT = 50000
PARALLEL_INPUT_CALLS = 16


# normalize dataset
def pre_process(image, label):
    return (image / 256).astype('float32'), tf.keras.utils.to_categorical(label, num_classes=100)


def generate_tf_data(X_train, y_train, X_test, y_test, batch_size, for_capsule=True):
    dataset_train = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    dataset_train = dataset_train.shuffle(buffer_size=CIFAR_TRAIN_IMAGE_COUNT)
    dataset_train = dataset_train.map(image_rotate_random)
    dataset_train = dataset_train.map(image_shift_rand,
                                      num_parallel_calls=PARALLEL_INPUT_CALLS)
    dataset_train = dataset_train.map(image_squish_random,
                                      num_parallel_calls=PARALLEL_INPUT_CALLS)
    dataset_train = dataset_train.map(image_erase_random,
                                      num_parallel_calls=PARALLEL_INPUT_CALLS)
    if for_capsule:
        dataset_train = dataset_train.map(generator,
                                          num_parallel_calls=PARALLEL_INPUT_CALLS)
    dataset_train = dataset_train.batch(batch_size)
    dataset_train = dataset_train.prefetch(-1)

    dataset_test = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    dataset_test = dataset_test.cache()
    if for_capsule:
        dataset_test = dataset_test.map(generator,
                                        num_parallel_calls=PARALLEL_INPUT_CALLS)
    dataset_test = dataset_test.batch(batch_size)
    dataset_test = dataset_test.prefetch(-1)

    return dataset_train, dataset_test