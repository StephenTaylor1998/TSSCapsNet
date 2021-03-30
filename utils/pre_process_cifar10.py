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

import numpy as np
import tensorflow as tf
import os
import cv2
from keras_preprocessing.image import ImageDataGenerator

tf2 = tf.compat.v2

# constants
CIFAR_IMG_SIZE = 32
CIFAR_IMG_CHANNEL = 3
CIFAR_TRAIN_IMAGE_COUNT = 50000
PARALLEL_INPUT_CALLS = 16


# normalize dataset
def pre_process(image, label):
    # print((image / 256).astype('float32'))
    return (image / 256).astype('float32'), tf.keras.utils.to_categorical(label, num_classes=10)


def image_shift_rand(image, label):
    image = tf.transpose(image, (2, 0, 1))
    image = tf.reshape(image, [CIFAR_IMG_SIZE, CIFAR_IMG_SIZE, CIFAR_IMG_CHANNEL])
    nonzero_x_cols = tf.cast(tf.where(tf.greater(
        tf.reduce_sum(image, axis=0), 0)), tf.int32)
    nonzero_y_rows = tf.cast(tf.where(tf.greater(
        tf.reduce_sum(image, axis=1), 0)), tf.int32)
    left_margin = tf.reduce_min(nonzero_x_cols)
    right_margin = CIFAR_IMG_SIZE - tf.reduce_max(nonzero_x_cols) - 1
    top_margin = tf.reduce_min(nonzero_y_rows)
    bot_margin = CIFAR_IMG_SIZE - tf.reduce_max(nonzero_y_rows) - 1
    rand_dirs = tf.random.uniform([2])
    dir_idxs = tf.cast(tf.floor(rand_dirs * 2), tf.int32)
    rand_amts = tf.minimum(tf.abs(tf.random.normal([2], 0, .33)), .9999)
    x_amts = [tf.floor(-1.0 * rand_amts[0] * tf.cast(left_margin, tf.float32)),
              tf.floor(rand_amts[0] * tf.cast(1 + right_margin, tf.float32))]
    y_amts = [tf.floor(-1.0 * rand_amts[1] * tf.cast(top_margin, tf.float32)),
              tf.floor(rand_amts[1] * tf.cast(1 + bot_margin, tf.float32))]
    x_amt = tf.cast(tf.gather(x_amts, dir_idxs[1], axis=0), tf.int32)
    y_amt = tf.cast(tf.gather(y_amts, dir_idxs[0], axis=0), tf.int32)
    image = tf.reshape(image, [CIFAR_IMG_SIZE * CIFAR_IMG_SIZE * CIFAR_IMG_CHANNEL])
    image = tf.roll(image, y_amt * CIFAR_IMG_SIZE, axis=0)
    image = tf.reshape(image, [CIFAR_IMG_SIZE, CIFAR_IMG_SIZE, CIFAR_IMG_CHANNEL])
    image = tf.transpose(image)
    image = tf.reshape(image, [CIFAR_IMG_SIZE * CIFAR_IMG_SIZE * CIFAR_IMG_CHANNEL])
    image = tf.roll(image, x_amt * CIFAR_IMG_SIZE, axis=0)
    # image = tf.reshape(image, [CIFAR_IMG_SIZE, CIFAR_IMG_SIZE, CIFAR_IMG_CHANNEL])
    image = tf.reshape(image, [CIFAR_IMG_CHANNEL, CIFAR_IMG_SIZE, CIFAR_IMG_SIZE])
    image = tf.transpose(image)
    # image = tf.reshape(image, [CIFAR_IMG_SIZE, CIFAR_IMG_SIZE, CIFAR_IMG_CHANNEL])
    image = tf.reshape(image, [CIFAR_IMG_CHANNEL, CIFAR_IMG_SIZE, CIFAR_IMG_SIZE])

    image = tf.transpose(image, (1, 2, 0))
    return image, label


def image_rotate_random_py_func(image, angle):
    rot_mat = cv2.getRotationMatrix2D(
        (CIFAR_IMG_SIZE / 2, CIFAR_IMG_SIZE / 2), int(angle), 1.0)
    # image = tf.squeeze(image, axis=-1)
    rotated = cv2.warpAffine(image.numpy(), rot_mat,
                             (CIFAR_IMG_SIZE, CIFAR_IMG_SIZE))
    return rotated


def image_rotate_random(image, label):
    rand_amts = tf.maximum(tf.minimum(
        tf.random.normal([2], 0, .33), .9999), -.9999)
    angle = rand_amts[0] * 30  # degrees
    new_image = tf.py_function(image_rotate_random_py_func,
                               (image, angle), tf.float32)
    new_image = tf.cond(rand_amts[1] > 0, lambda: image, lambda: new_image)
    return new_image, label


def image_erase_random(image, label):
    sess = tf.compat.v1.Session()
    with sess.as_default():
        rand_amts = tf.random.uniform([2])
        x = tf.cast(tf.floor(rand_amts[0] * 19) + 4, tf.int32)
        y = tf.cast(tf.floor(rand_amts[1] * 19) + 4, tf.int32)
        patch = tf.zeros([4, 4])
        mask = tf.pad(patch, [[x, CIFAR_IMG_SIZE - x - 4],
                              [y, CIFAR_IMG_SIZE - y - 4]],
                      mode='CONSTANT', constant_values=1)
        image = tf.multiply(image, tf.expand_dims(mask, -1))
        return image, label


def image_squish_random(image, label):
    rand_amts = tf.minimum(tf.abs(tf.random.normal([2], 0, .33)), .9999)
    width_mod = tf.cast(tf.floor(
        (rand_amts[0] * (CIFAR_IMG_SIZE / 4)) + 1), tf.int32)
    offset_mod = tf.cast(tf.floor(rand_amts[1] * 2.0), tf.int32)
    offset = (width_mod // 2) + offset_mod
    print(image)
    print(type(image))
    image = tf.image.resize(image,
                            [CIFAR_IMG_SIZE, CIFAR_IMG_SIZE - width_mod],
                            method=tf2.image.ResizeMethod.LANCZOS3,
                            preserve_aspect_ratio=False,
                            antialias=True)
    image = tf.image.pad_to_bounding_box(
        image, 0, offset, CIFAR_IMG_SIZE, CIFAR_IMG_SIZE + offset_mod)
    image = tf.image.crop_to_bounding_box(
        image, 0, 0, CIFAR_IMG_SIZE, CIFAR_IMG_SIZE)
    return image, label


def generator(image, label):
    return (image, label), (label, image)


# datagen = ImageDataGenerator(
#     # set input mean to 0 over the dataset
#     featurewise_center=False,
#     # set each sample mean to 0
#     samplewise_center=False,
#     # divide inputs by std of dataset
#     featurewise_std_normalization=False,
#     # divide each input by its std
#     samplewise_std_normalization=False,
#     # apply ZCA whitening
#     zca_whitening=False,
#     # epsilon for ZCA whitening
#     zca_epsilon=1e-06,
#     # randomly rotate images in the range (deg 0 to 180)
#     rotation_range=0,
#     # randomly shift images horizontally
#     width_shift_range=0.1,
#     # randomly shift images vertically
#     height_shift_range=0.1,
#     # set range for random shear
#     shear_range=0.,
#     # set range for random zoom
#     zoom_range=0.,
#     # set range for random channel shifts
#     channel_shift_range=0.,
#     # set mode for filling points outside the input boundaries
#     fill_mode='nearest',
#     # value used for fill_mode = "constant"
#     cval=0.,
#     # randomly flip images
#     horizontal_flip=True,
#     # randomly flip images
#     vertical_flip=False,
#     # set rescaling factor (applied before any other transformation)
#     rescale=None,
#     # set function that will be applied on each input
#     preprocessing_function=None,
#     # image data format, either "channels_first" or "channels_last"
#     data_format=None,
#     # fraction of images reserved for validation (strictly between 0 and 1)
#     validation_split=0.0)

# Compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied).
# datagen.fit(x_train)

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
