import numpy as np
import tensorflow as tf

from ..layers.transform.dwt import DWT
from ..layers.layers_efficient import PrimaryCaps, FCCaps, Length, Mask, generator_graph_mnist


def dwt_capsnet_graph(input_shape, num_classes, name):
    """
    Efficient-CapsNet graph architecture.

    Parameters
    ----------
    :param input_shape:
    :param num_classes:
    :param name:
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
    # (18, 18, 32) ==>> (9, 9, 128)
    x = DWT()(x)

    x = tf.keras.layers.BatchNormalization()(x)
    x = PrimaryCaps(128, x.shape[1], 16, 8)(x)

    digit_caps = FCCaps(num_classes, 16)(x)

    digit_caps_len = Length(name='length_capsnet_output')(digit_caps)

    return tf.keras.Model(inputs=inputs, outputs=[digit_caps, digit_caps_len], name=name)


def build_graph(input_shape, mode, name, num_classes=10):
    inputs = tf.keras.Input(input_shape)
    y_true = tf.keras.layers.Input(shape=(num_classes,))
    noise = tf.keras.layers.Input(shape=(num_classes, 16))

    efficient_capsnet = dwt_capsnet_graph(input_shape, num_classes, name)

    efficient_capsnet.summary()
    print("\n\n")

    digit_caps, digit_caps_len = efficient_capsnet(inputs)
    noised_digitcaps = tf.keras.layers.Add()([digit_caps, noise])  # only if mode is play

    masked_by_y = Mask()([digit_caps, y_true])
    masked = Mask()(digit_caps)
    masked_noised_y = Mask()([noised_digitcaps, y_true])

    generator = generator_graph_mnist(input_shape)

    generator.summary()
    print("\n\n")

    x_gen_train = generator(masked_by_y)
    x_gen_eval = generator(masked)
    x_gen_play = generator(masked_noised_y)

    if mode == 'train':
        return tf.keras.models.Model([inputs, y_true], [digit_caps_len, x_gen_train],
                                     name='DWT_Efficinet_CapsNet_Generator')
    elif mode == 'test':
        return tf.keras.models.Model(inputs, [digit_caps_len, x_gen_eval], name='DWT_Efficinet_CapsNet_Generator')
    elif mode == 'play':
        return tf.keras.models.Model([inputs, y_true, noise], [digit_caps_len, x_gen_play],
                                     name='DWT_Efficinet_CapsNet_Generator')
    else:
        raise RuntimeError('mode not recognized')
