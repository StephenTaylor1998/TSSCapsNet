import tensorflow as tf


def block2channel_2d(tf_tensor_2d, block_shape, output_channel_first=False, check_shape=True):
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
    :param tf_tensor_2d: input tf tensor ([h, w] or [batch, h, w])
    (Attention: support batch, like shape[batch, h, w])
    :param block_shape: block shape should < & = input tensor shape
    :param output_channel_first: channel first ==>> True & channel last ==>> False;
    :param check_shape: check shape before operator
    :return: 3d tf tensor
    """
    tensor_h = tf_tensor_2d.shape[-2]
    tensor_w = tf_tensor_2d.shape[-1]

    block_h = block_shape[-2]
    block_w = block_shape[-1]

    if check_shape:
        assert tensor_h % block_h == 0, \
            "check the tensor_h and block_h, the func recommend tensor_h % block_h == 0"
        assert tensor_w % block_w == 0, \
            "check the tensor_w and block_w, the func recommend tensor_w % block_w == 0"

    t1 = tf.reshape(tf_tensor_2d, (None, tensor_h // block_h, block_h, tensor_w // block_w, block_w))

    if output_channel_first:
        t2 = tf.transpose(t1, (None, -3, -1, -4, -2))
        out = tf.reshape(t2, (None, block_h * block_w, tensor_h // block_h, tensor_w // block_w))
    else:
        t2 = tf.transpose(t1, (None, -4, -2, -3, -1))
        out = tf.reshape(t2, (None, tensor_h // block_h, tensor_w // block_w, block_h * block_w))

    return out


def dct2channel(tf_tensor_2d, block_shape, output_channel_first=False, check_shape=True):
    out = block2channel_2d(tf_tensor_2d, block_shape, output_channel_first, check_shape)
    return tf.signal.dct(out.astype(tf.float32))
