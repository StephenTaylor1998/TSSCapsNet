import numpy as np
import tensorflow as tf
from scipy import fft
from .block2channel import Block2Channel2d
from .block2channel import Block2Channel3d
from .block2channel import block2channel_2d
from .block2channel import block2channel_3d


def dct2channel(np_array, block_shape, check_shape=True):
    """
    Convert 2d or 3d numpy array to 3d, then do DCT op at last dimension.
    :param np_array: 2d or 3d array
    :param block_shape: (block_h, block_w) example (2, 2),
    block shape should <= input tensor shape
    :param check_shape: check shape while run block2channel_2d(...) or block2channel_3d(...)
    :return: 3d numpy array, the same like output of block2channel_2d(...) or block2channel_3d(...)
    """
    if len(np_array.shape) == 2:
        out = block2channel_2d(np_array, block_shape, False, check_shape)

    elif len(np_array.shape) == 3:
        out = block2channel_3d(np_array, block_shape, False, check_shape)

    else:
        print("shape {} not support, recommend [h, w] or [h, w, channel]".format(np_array.shape))
        raise NotImplementedError

    return fft.dct(out.astype(np.float32))


class DCTLayer2d(tf.keras.layers.Layer):
    """
    Convert tf tensor with batch(like [batch, h, w]) to [h//block_H, w//block_w, block_h*block_w]
    then do DCT op at last dimension.
    :param block_shape: (block_h, block_w) example (2, 2),
    block shape should <= input tensor shape
    :param check_shape: check shape while run block2channel_2d(...)
    :return: [batch, h//block_H, w//block_w, block_h*block_w]
    """

    def __init__(self, block_shape, check_shape=True, **kwargs):
        super(DCTLayer2d, self).__init__(**kwargs)
        self.block_shape = block_shape
        self.check_shape = check_shape
        self.block2Channel2d = None

    def build(self, input_shape):
        self.block2Channel2d = Block2Channel2d(self.block_shape, False, self.check_shape)

    def call(self, inputs, **kwargs):
        # [batch, h, w] ==>> [batch, h//block_H, w//block_w, block_h*block_w]
        out = self.block2Channel2d(inputs)
        return tf.signal.dct(tf.cast(out, dtype=tf.float32))


class DCTLayer3d(tf.keras.layers.Layer):
    """
    Convert tf tensor with batch [batch, h, w, channel] to [batch, h//block_H, w//block_w, channel*block_h*block_w],
    then do DCT op at last dimension.
    :param block_shape: (block_h, block_w) example (2, 2),
    block shape should <= input tensor shape
    :param dct_type: default 2, example tf.signal.dct(tensor, type=dct_type)
    :param check_shape: check shape while run block2channel_3d(...)
    :return: [batch, h//block_H, w//block_w, channel*block_h*block_w]
    """

    def __init__(self, block_shape, groups=None, dct_type=2, check_shape=True, **kwargs):
        super(DCTLayer3d, self).__init__(**kwargs)
        self.block_shape = block_shape
        self.groups = groups
        self.dct_type = dct_type
        self.check_shape = check_shape
        self.block2Channel3d = None

    def build(self, input_shape):
        self.block2Channel3d = Block2Channel3d(self.block_shape, False, self.check_shape)

    def call(self, inputs, **kwargs):
        # [batch, h, w, channel] ==>> [batch, h//block_H, w//block_w, channel*block_h*block_w]
        out = self.block2Channel3d(inputs)
        if self.groups:
            print("NotImplemented!")
            raise NotImplemented

        else:
            return tf.signal.dct(tf.cast(out, dtype=tf.float32), self.dct_type)
