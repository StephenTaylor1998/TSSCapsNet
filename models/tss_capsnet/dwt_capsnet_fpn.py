import numpy as np
import tensorflow as tf
from models.layers import RoutingA
from models.layers.layers_efficient import PrimaryCaps, Length, Mask
from models.layers.operators import Heterogeneous
from models.layers.routing import Routing
from models.layers.transform import DWT


def efficient_capsnet_graph(input_shape, num_classes, routing_name_list, regularize=1e-4):
    """
    reimplement for cifar dataset
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
    x = PrimaryCaps(128, x.shape[1], 16, 8)(x)

    digit_caps = Routing(num_classes, routing_name_list, regularize)(x)

    # x = layers.LayerNormalization()(x)
    # digit_caps = FCCaps(10, 16)(x)

    digit_caps_len = Length(name='length_capsnet_output')(digit_caps)

    digit_caps_len = Heterogeneous(num_class=10)((x, digit_caps_len))

    # digit_caps_len = tf.keras.layers.Softmax()(digit_caps_len)

    return tf.keras.Model(inputs=inputs, outputs=[digit_caps, digit_caps_len], name='DWT_Multi_Attention_CapsNet')


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


def build_graph(input_shape, mode, num_classes, routing_name_list, regularize=1e-4):
    """
    Efficient-CapsNet graph architecture with reconstruction regularizer.
    The network can be initialize with different modalities.

    Parameters
    ----------
    :param input_shape: network input shape
    :param mode: working mode ('train', 'test' & 'play')
    :param input_shape:
    :param num_classes:
    :param routing_name_list:
    :param regularize:
    """
    inputs = tf.keras.Input(input_shape)
    y_true = tf.keras.layers.Input(shape=(10,))
    noise = tf.keras.layers.Input(shape=(10, 16))

    efficient_capsnet = efficient_capsnet_graph(input_shape, num_classes, routing_name_list, regularize)
    efficient_capsnet.summary()

    digit_caps, digit_caps_len = efficient_capsnet(inputs)
    noised_digitcaps = tf.keras.layers.Add()([digit_caps, noise])  # only if mode is play

    masked_by_y = Mask()([digit_caps, y_true])
    masked = Mask()(digit_caps)
    masked_noised_y = Mask()([noised_digitcaps, y_true])

    generator = generator_graph(input_shape)


    generator.summary()


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
