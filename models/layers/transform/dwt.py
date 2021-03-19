import pywt
import numpy as np
import tensorflow as tf


class DWT(tf.keras.layers.Layer):
    """
    Convert tf tensor with DWT transform.
    shape[batch, h, w, channel] to shape[batch, h//2, w//2, channel*4].

    :param wave_name: input numpy array([h, w, channel])
    (Waring: Do not support batch, like shape[batch, h, w, channel])
    :param strides: (block_h, block_w) example (2, 2),
    block shape should <= input tensor shape
    :return: 3d tensor [batch, h//2, w//2, channel*4].
    """

    def __init__(self, wave_name='haar', strides=None):
        super(DWT, self).__init__()
        wavelet = pywt.Wavelet(wave_name)
        if strides is None:
            self.strides = [1, 1, 2, 2, 1]
        else:
            self.strides = strides

        # shape (2) ==>> (2, 2) ==>> (2, 2, 1)
        f1 = np.expand_dims(np.outer(wavelet.dec_lo, wavelet.dec_lo), axis=-1)
        f2 = np.expand_dims(np.outer(wavelet.dec_hi, wavelet.dec_lo), axis=-1)
        f3 = np.expand_dims(np.outer(wavelet.dec_lo, wavelet.dec_hi), axis=-1)
        f4 = np.expand_dims(np.outer(wavelet.dec_hi, wavelet.dec_hi), axis=-1)
        # shape 4*(2, 2, 1) ==>> (2, 2, 4)
        filters = np.concatenate((f1, f2, f3, f4), axis=-1)[::-1, ::-1]
        # shape 4*(2, 2, 4) ==>> (1, 2, 2, 1, 4)
        filters = np.expand_dims(filters, axis=(0, -2))
        self.filter = tf.Variable(filters, trainable=False, dtype=tf.float32)
        self.size = 2 * (len(wavelet.dec_lo) // 2 - 1)

    def build(self, input_shape):

        self.built = True

    def call(self, inputs, **kwargs):
        x = tf.pad(inputs,
                   tf.constant([[0, 0],
                                [self.size, self.size],
                                [self.size, self.size],
                                [0, 0]]),
                   mode='reflect')
        x = tf.expand_dims(x, 0)
        x = tf.transpose(x, (0, 4, 2, 3, 1))
        x = tf.nn.conv3d(x, self.filter, padding='VALID', strides=self.strides)
        x = tf.transpose(x, (0, 2, 3, 1, 4))
        shape = tf.shape(x)
        x = tf.reshape(x, (shape[0], shape[1], shape[2], shape[3]*shape[4]))
        return x
