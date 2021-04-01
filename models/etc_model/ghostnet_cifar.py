import math
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Concatenate, DepthwiseConv2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Dropout, BatchNormalization, Activation, Reshape


def ghost_conv_2d(x, out_channels, ratio, l1_kernel_size, l2_kernel_size, padding='same', strides=1, data_format='channels_last',
                  use_bias=False, activation=None):
    l1_channel = math.ceil(out_channels * 1.0 / ratio)
    x = Conv2D(int(l1_channel), (l1_kernel_size, l1_kernel_size), strides=(strides, strides), padding=padding,
               data_format=data_format, activation=activation, use_bias=use_bias)(x)
    if ratio == 1:
        return x
    dw = DepthwiseConv2D(l2_kernel_size, strides, padding=padding, depth_multiplier=ratio - 1, data_format=data_format,
                         activation=activation, use_bias=use_bias)(x)

    x = Concatenate(axis=-1)([x, dw])
    return x


def squeeze_excitation_block(x, outchannels, ratio):
    x1 = GlobalAveragePooling2D(data_format='channels_last')(x)
    squeeze = Reshape((1, 1, int(x1.shape[-1])))(x1)
    fc1 = Conv2D(int(outchannels / ratio), (1, 1), strides=(1, 1), padding='same', data_format='channels_last',
                 use_bias=False, activation=None)(squeeze)
    relu = Activation('relu')(fc1)
    fc2 = Conv2D(int(outchannels), (1, 1), strides=(1, 1), padding='same', data_format='channels_last',
                 use_bias=False, activation=None)(relu)
    excitation = Activation('hard_sigmoid')(fc2)
    scale = excitation * x
    return scale


def MobileNetBottleneck(x, dwkernel, strides, exp, out, ratio, use_se):
    x1 = DepthwiseConv2D(dwkernel, strides, padding='same', depth_multiplier=ratio - 1, data_format='channels_last',
                         activation=None, use_bias=False)(x)
    x1 = BatchNormalization(axis=-1)(x1)
    x1 = Conv2D(out, (1, 1), strides=(1, 1), padding='same', data_format='channels_last',
                activation=None, use_bias=False)(x1)
    x1 = BatchNormalization(axis=-1)(x1)
    y = ghost_conv_2d(x, exp, ratio, 1, 3)
    y = BatchNormalization(axis=-1)(y)
    y = Activation('relu')(y)
    if strides > 1:
        y = DepthwiseConv2D(dwkernel, strides, padding='same', depth_multiplier=ratio - 1, data_format='channels_last',
                            activation=None, use_bias=False)(y)
        y = BatchNormalization(axis=-1)(y)
        y = Activation('relu')(y)
    if use_se:
        y = squeeze_excitation_block(y, exp, ratio)
    y = ghost_conv_2d(y, out, ratio, 1, 3)
    y = BatchNormalization(axis=-1)(y)
    y = tf.keras.layers.add([x1, y])
    return y


def build_graph(input_shape, num_classes=10):
    inputdata = tf.keras.Input(shape=input_shape)

    x = Conv2D(16, (3, 3), strides=(2, 2), padding='same', data_format='channels_last', activation=None,
               use_bias=False)(inputdata)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)

    x = MobileNetBottleneck(x, 3, 1, 16, 16, 2, False)
    x = MobileNetBottleneck(x, 3, 2, 48, 24, 2, False)
    x = MobileNetBottleneck(x, 3, 1, 72, 24, 2, False)
    x = MobileNetBottleneck(x, 5, 2, 72, 40, 2, True)
    x = MobileNetBottleneck(x, 5, 1, 120, 40, 2, True)
    x = MobileNetBottleneck(x, 3, 2, 240, 80, 2, False)
    x = MobileNetBottleneck(x, 3, 1, 200, 80, 2, False)
    x = MobileNetBottleneck(x, 3, 1, 184, 80, 2, False)
    x = MobileNetBottleneck(x, 3, 1, 184, 80, 2, False)
    x = MobileNetBottleneck(x, 3, 1, 480, 112, 2, True)
    x = MobileNetBottleneck(x, 3, 1, 672, 112, 2, True)
    x = MobileNetBottleneck(x, 5, 2, 672, 160, 2, True)
    x = MobileNetBottleneck(x, 5, 1, 960, 160, 2, False)
    x = MobileNetBottleneck(x, 5, 1, 960, 160, 2, True)
    x = MobileNetBottleneck(x, 5, 1, 960, 160, 2, False)
    x = MobileNetBottleneck(x, 5, 1, 960, 160, 2, True)

    x = Conv2D(960, (1, 1), strides=(1, 1), padding='same', data_format='channels_last', activation=None,
               use_bias=False)(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)

    x = GlobalAveragePooling2D(data_format='channels_last')(x)
    x = Reshape((1, 1, int(x.shape[-1])))(x)
    x = Conv2D(1280, (1, 1), strides=(1, 1), padding='same', data_format='channels_last', activation=None,
               use_bias=False)(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)
    x = Dropout(0.05)(x)
    x = Conv2D(num_classes, (1, 1), strides=(1, 1), padding='same', data_format='channels_last', activation=None,
               use_bias=False)(x)
    x = tf.squeeze(x, 1)
    x = tf.squeeze(x, 1)

    out = tf.keras.layers.Softmax(name="GHOSTNET")(x)

    model = tf.keras.Model(inputdata, out)
    model.summary()
    return model
