import numpy as np
import tensorflow as tf
from ..layers import DCTLayer3d, GumbelGate
from ..layers.gate import DynamicGate
from ..layers.layers_hinton import PrimaryCaps, DigitCaps, Length, Mask, generator_graph_hinton_mnist


def dct_capsnet_graph(input_shape, routing, name):

    inputs = tf.keras.Input(input_shape)
    # (28, 28, 1) ==>> (26, 26, 16)
    x = tf.keras.layers.Conv2D(16, 3, activation="relu", padding='valid', kernel_initializer='he_normal')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    # (26, 26, 16) ==>> (24, 24, 16)
    x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    # (24, 24, 16) ==>> (22, 22, 32)
    x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    # (22, 22, 32) ==>> (20, 20, 16)
    x = tf.keras.layers.Conv2D(16, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    # (20, 20, 16) ==>> (10, 10, 64)
    x = DCTLayer3d(block_shape=(2, 2))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    # x, key = GumbelGate(act='relu')(x, temperature=1e-3)
    x, key = DynamicGate(act='relu')(x, temperature=1e-3)
    # (10, 10, 64) == >> (10, 10, 256) ==>> (6, 6, 32, 8)
    primary = PrimaryCaps(C=32, L=8, k=5, s=1)(x)
    digit_caps = DigitCaps(10, 16, routing=routing)(primary)
    digit_caps_len = Length(name='capsnet_output_len')(digit_caps)
    pr_shape = primary.shape
    primary = tf.reshape(primary, (-1, pr_shape[1] * pr_shape[2] * pr_shape[3], pr_shape[-1]))

    return tf.keras.Model(inputs=inputs, outputs=[primary, digit_caps, digit_caps_len], name=name)


def build_graph(input_shape, mode, n_routing, name):
    inputs = tf.keras.Input(input_shape)
    y_true = tf.keras.Input(shape=(10))
    noise = tf.keras.layers.Input(shape=(10, 16))

    capsnet = dct_capsnet_graph(input_shape, routing=n_routing, name=name)
    primary, digit_caps, digit_caps_len = capsnet(inputs)
    noised_digitcaps = tf.keras.layers.Add()([digit_caps, noise])  # only if mode is play

    capsnet.summary()
    print("\n\n")

    masked_by_y = Mask()(
        [digit_caps, y_true])  # The true label is used to mask the output of capsule layer. For training
    masked = Mask()(digit_caps)  # Mask using the capsule with maximal length. For prediction
    masked_noised_y = Mask()([noised_digitcaps, y_true])

    generator = generator_graph_hinton_mnist(input_shape)

    generator.summary()
    print("\n\n")

    x_gen_train = generator(masked_by_y)
    x_gen_eval = generator(masked)
    x_gen_play = generator(masked_noised_y)

    if mode == 'train':
        return tf.keras.models.Model([inputs, y_true], [digit_caps_len, x_gen_train], name='DCT_CapsNet_Generator')
    elif mode == 'test':
        return tf.keras.models.Model(inputs, [digit_caps_len, x_gen_eval], name='DCT_CapsNet_Generator')
    elif mode == 'play':
        return tf.keras.models.Model([inputs, y_true, noise], [digit_caps_len, x_gen_play], name='DCT_CapsNet_Generator')
    else:
        raise RuntimeError('mode not recognized')

