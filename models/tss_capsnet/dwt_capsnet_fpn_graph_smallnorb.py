import tensorflow as tf
import tensorflow_addons as tfa
from ..layers.layers_efficient import PrimaryCaps, FCCaps, Length, Mask, generator_graph_smallnorb
from ..layers.routing import Routing
from ..layers.transform import DWT


def dwt_capsnet_graph(input_shape, num_classes=10, routing_name_list=None, regularize=1e-4, name=None):
    """
    Efficient-CapsNet graph architecture.
    
    Parameters
    ----------   
    input_shape: list
        network input shape
        :param name:
        :param regularize:
        :param routing_name_list:
        :param num_classes:
    """
    inputs = tf.keras.Input(input_shape)

    x = tf.keras.layers.Conv2D(32, 7, 2, activation=None, padding='valid', kernel_initializer='he_normal')(inputs)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tfa.layers.InstanceNormalization(axis=3,
                                         center=True,
                                         scale=True,
                                         beta_initializer="random_uniform",
                                         gamma_initializer="random_uniform")(x)
    x = tf.keras.layers.Conv2D(64, 3, activation=None, padding='valid', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tfa.layers.InstanceNormalization(axis=3,
                                         center=True,
                                         scale=True,
                                         beta_initializer="random_uniform",
                                         gamma_initializer="random_uniform")(x)
    x = tf.keras.layers.Conv2D(64, 3, activation=None, padding='valid', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tfa.layers.InstanceNormalization(axis=3,
                                         center=True,
                                         scale=True,
                                         beta_initializer="random_uniform",
                                         gamma_initializer="random_uniform")(x)
    x = tf.keras.layers.Conv2D(32, 3, 2, activation=None, padding='valid', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tfa.layers.InstanceNormalization(axis=3,
                                         center=True,
                                         scale=True,
                                         beta_initializer="random_uniform",
                                         gamma_initializer="random_uniform")(x)
    x = DWT()(x)

    x = PrimaryCaps(128, 8, 16, 8)(x)  # there could be an error

    digit_caps = Routing(num_classes, routing_name_list, regularize)(x)
    digit_caps_len = Length(name='length_capsnet_output')(digit_caps)
    return tf.keras.Model(inputs=inputs, outputs=[digit_caps, digit_caps_len], name=name)


def build_graph(input_shape, mode, num_classes, routing_name_list, regularize=1e-4, name=''):
    inputs = tf.keras.Input(input_shape)
    y_true = tf.keras.layers.Input(shape=(5,))

    efficient_capsnet = dwt_capsnet_graph(input_shape, num_classes, routing_name_list, regularize, name)

    efficient_capsnet.summary()
    print("\n\n")

    digit_caps, digit_caps_len = efficient_capsnet(inputs)

    masked_by_y = Mask()([digit_caps, y_true])
    masked = Mask()(digit_caps)

    generator = generator_graph_smallnorb(input_shape)

    generator.summary()
    print("\n\n")

    x_gen_train = generator(masked_by_y)
    x_gen_eval = generator(masked)

    if mode == 'train':
        return tf.keras.models.Model([inputs, y_true], [digit_caps_len, x_gen_train])
    elif mode == 'test':
        return tf.keras.models.Model(inputs, [digit_caps_len, x_gen_eval])
    else:
        raise RuntimeError('mode not recognized')
