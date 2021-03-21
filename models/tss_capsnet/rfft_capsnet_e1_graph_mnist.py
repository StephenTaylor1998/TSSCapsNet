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

import numpy as np
import tensorflow as tf

from ..layers import RFFTLayer3d
from ..layers.layers_efficient import PrimaryCaps, FCCaps, Length, Mask


def efficient_capsnet_graph(input_shape):
    """
    Efficient-CapsNet graph architecture.

    Parameters
    ----------
    input_shape: list
        network input shape
    """
    inputs = tf.keras.Input(input_shape)
    # (28, 28, 1) ==>> (24, 24, 32)
    x = tf.keras.layers.Conv2D(32, 5, activation="relu", padding='valid', kernel_initializer='he_normal')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    # (24, 24, 32) ==>> (22, 22, 64)
    x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    # (22, 22, 64) ==>> (20, 20, 64)
    x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    # (20, 20, 64) ==>> (18, 18, 32)
    x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    # (18, 18, 32) ==>> (9, 9, 65)
    x = RFFTLayer3d(block_shape=(2, 2))(x)
    x = x[:, :, :, :-1]
    x = tf.keras.layers.BatchNormalization()(x)
    # x = PrimaryCaps(65, 9, 13, 5)(x)
    x = PrimaryCaps(64, 9, 16, 4)(x)

    digit_caps = FCCaps(10, 16)(x)
    # digit_caps = FCCaps(10, 13)(x)

    digit_caps_len = Length(name='length_capsnet_output')(digit_caps)

    return tf.keras.Model(inputs=inputs, outputs=[digit_caps, digit_caps_len], name='RFFT_Efficient_CapsNet')


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
                                     name='RFFT_Efficient_CapsNet_Generator')
    elif mode == 'test':
        return tf.keras.models.Model(inputs, [digit_caps_len, x_gen_eval], name='RFFT_Efficient_CapsNet_Generator')
    elif mode == 'play':
        return tf.keras.models.Model([inputs, y_true, noise], [digit_caps_len, x_gen_play],
                                     name='RFFT_Efficient_CapsNet_Generator')
    else:
        raise RuntimeError('mode not recognized')
