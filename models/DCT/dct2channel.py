import numpy as np
import tensorflow as tf


def block2channel_np(np_array_2d, block_shape, channel_first=True, check_shape=True):
    """
    example:
    block_shape=(2, 2)
    channel_first=False
    np_array_2d=
    [1, 2,  1, 2,  1, 2],
    [3, 4,  3, 4,  3, 4],

    [1, 2,  1, 2,  1, 2],
    [3, 4,  3, 4,  3, 4],

    [1, 2,  1, 2,  1, 2],
    [3, 4,  3, 4,  3, 4],

    output=
          /\-------\
        / 1 \2 3 4  \
      / 1  1 \2 3 4  \
    / 1  1  1/ 2 3 4 /
    \  1  1/ 2 3 4 /
     \  1/ 2 3 4 /
      \/-------/
    :param np_array_2d: input numpy array
    :param block_shape: block shape should < & = input array shape
    :param channel_first: channel first ==>> True & channel last ==>> False;
    :param check_shape: check shape before operator
    :return: 3d numpy array
    """
    array_h = np_array_2d.shape[0]
    array_w = np_array_2d.shape[-1]

    block_h = block_shape[0]
    block_w = block_shape[-1]

    if check_shape:
        assert array_h % block_h == 0, \
            "check the array_h and block_h, the func recommend array_h % block_h == 0"
        assert array_w % block_w == 0, \
            "check the array_w and block_w, the func recommend array_w % block_w == 0"

    t1 = np.reshape(np_array_2d, (array_h // block_h, block_h, array_w // block_w, block_w))

    if channel_first:
        t2 = np.transpose(t1, (1, 3, 0, 2))
        out = np.reshape(t2, (block_h * block_w, array_h // block_h, array_w // block_w))
    else:
        t2 = np.transpose(t1, (0, 2, 1, 3))
        out = np.reshape(t2, (array_h // block_h, array_w // block_w, block_h * block_w))

    return out


def block2channel(tf_tensor_2d, block_shape, channel_first=True, check_shape=True):
    """
    example:
    block_shape=(2, 2)
    channel_first=False
    tf_tensor_2d=
    [1, 2,  1, 2,  1, 2],
    [3, 4,  3, 4,  3, 4],

    [1, 2,  1, 2,  1, 2],
    [3, 4,  3, 4,  3, 4],

    [1, 2,  1, 2,  1, 2],
    [3, 4,  3, 4,  3, 4],

    output=
          /\-------\
        / 1 \2 3 4  \
      / 1  1 \2 3 4  \
    / 1  1  1/ 2 3 4 /
    \  1  1/ 2 3 4 /
     \  1/ 2 3 4 /
      \/-------/
    :param tf_tensor_2d: input tf tensor
    :param block_shape: block shape should < & = input tensor shape
    :param channel_first: channel first ==>> True & channel last ==>> False;
    :param check_shape: check shape before operator
    :return: 3d tf tensor
    """
    tensor_h = tf_tensor_2d.shape[0]
    tensor_w = tf_tensor_2d.shape[-1]

    block_h = block_shape[0]
    block_w = block_shape[-1]

    if check_shape:
        assert tensor_h % block_h == 0, \
            "check the tensor_h and block_h, the func recommend tensor_h % block_h == 0"
        assert tensor_w % block_w == 0, \
            "check the tensor_w and block_w, the func recommend tensor_w % block_w == 0"

    t1 = tf.reshape(tf_tensor_2d, (tensor_h // block_h, block_h, tensor_w // block_w, block_w))

    if channel_first:
        t2 = tf.transpose(t1, (1, 3, 0, 2))
        out = tf.reshape(t2, (block_h * block_w, tensor_h // block_h, tensor_w // block_w))
    else:
        t2 = tf.transpose(t1, (0, 2, 1, 3))
        out = tf.reshape(t2, (tensor_h // block_h, tensor_w // block_w, block_h * block_w))

    return out



