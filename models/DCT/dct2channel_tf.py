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
    :param block_shape: (block_h, block_w) example (2, 2),
    block shape should <= input tensor shape
    :param output_channel_first: channel first ==>> True & channel last ==>> False;
    :param check_shape: check shape before operator
    :return: 3d tf tensor
    """
    tensor_h = tf_tensor_2d.shape[-2]
    tensor_w = tf_tensor_2d.shape[-1]

    block_h = block_shape[0]
    block_w = block_shape[-1]

    if check_shape:
        assert tensor_h % block_h == 0, \
            "check the tensor_h and block_h, the func recommend tensor_h % block_h == 0"
        assert tensor_w % block_w == 0, \
            "check the tensor_w and block_w, the func recommend tensor_w % block_w == 0"

    t1 = tf.reshape(tf_tensor_2d, (tensor_h // block_h, block_h, tensor_w // block_w, block_w))

    if output_channel_first:
        t2 = tf.transpose(t1, (1, 3, 0, 2))
        out = tf.reshape(t2, (block_h * block_w, tensor_h // block_h, tensor_w // block_w))
    else:
        t2 = tf.transpose(t1, (0, 2, 1, 3))
        out = tf.reshape(t2, (tensor_h // block_h, tensor_w // block_w, block_h * block_w))

    return out


def block2channel_3d(tf_tensor_3d, block_shape, output_channel_first=False, check_shape=True):
    """
    example:
    block_shape=(2, 2)
    channel_first=False
    tf_tensor_2d=
            /\-------------/\
          / 1 \----------/-5 \
        / 3  2 \-------/-7--6 \
      / 1  4  1 \----/-5--8--5 \
    / 3  2  3  2/--/-7--6--7--6/
    \  4  1  4/----\--8--5--8/
     \  3  2/-------\--7--6/
      \  4/----------\--8/
       \/-------------\/

    output=
        /\------\--------\
      / 1 \2 3 4 \5 6 7 8 \
    / 1  1/ 2 3 4/ 5 6 7 8/
    \  1/ 2 3 4/ 5 6 7 8/
     \/------/--------/
    :param tf_tensor_3d: input tf tensor ([h, w, channel] or [batch, h, w, channel])
    (Attention: support batch, like shape[batch, h, w, channel])
    :param block_shape: (block_h, block_w) example (2, 2),
    block shape should <= input tensor shape
    :param output_channel_first: channel first ==>> True & channel last ==>> False;
    :param check_shape: check shape before operator
    :return: 3d tf tensor
    """
    tensor_h = tf_tensor_3d.shape[-3]
    tensor_w = tf_tensor_3d.shape[-2]
    tensor_c = tf_tensor_3d.shape[-1]

    block_h = block_shape[0]
    block_w = block_shape[-1]

    if check_shape:
        assert len(tf_tensor_3d.shape) == 3, \
            "check the dims of np_array_2d, the func recommend len(tf_tensor_3d.shape) == 3"
        assert tensor_h % block_h == 0, \
            "check the tensor_h and block_h, the func recommend tensor_h % block_h == 0"
        assert tensor_w % block_w == 0, \
            "check the tensor_w and block_w, the func recommend tensor_w % block_w == 0"

    t1 = tf.reshape(tf_tensor_3d, (tensor_h // block_h, block_h, tensor_w // block_w, block_w, tensor_c))

    if output_channel_first:
        t2 = tf.transpose(t1, (0, 2, 4, 1, 3))
        out = tf.reshape(t2, (tensor_c * block_h * block_w, tensor_h // block_h, tensor_w // block_w))
    else:
        t2 = tf.transpose(t1, (1, 3, 0, 2, 4))
        out = tf.reshape(t2, (tensor_h // block_h, tensor_w // block_w, tensor_c * block_h * block_w))

    return out


def dct2channel_from_2d(tf_tensor_2d, block_shape, check_shape=True):
    """
    Convert 2d tf tensor(like [h, w]) to 3d(like [h, w, channel]) or,
    2d tf tensor with batch(like [batch, h, w]) to 3d(like [batch, h, w, channel])
    then do DCT op at last dimension.
    :param tf_tensor_2d: 2d tf tensor(like [h, w]) or 2d tf tensor with batch(like [batch, h, w])
    :param block_shape: (block_h, block_w) example (2, 2),
    block shape should <= input tensor shape
    :param check_shape: check shape while run block2channel_2d(...) or block2channel_3d(...)
    :return: [h//block_H, w//block_w, block_h*block_w]
    """
    out = block2channel_2d(tf_tensor_2d, block_shape, False, check_shape)
    return tf.signal.dct(tf.cast(out, dtype=tf.float32))


def dct2channel_from_3d(tf_tensor_3d, block_shape, check_shape=True):
    """
    Convert 3d tf tensor(like [h, w, channel]) to 3d(like [h, w, channel]) or,
    3d tf tensor with batch(like [batch, h, w, channel]) to 3d(like [batch, h, w, channel])
    then do DCT op at last dimension.
    :param tf_tensor_3d: 3d tf tensor(like [h, w, channel]) or
     3d tf tensor with batch(like [batch, h, w, channel])
    :param block_shape: (block_h, block_w) example (2, 2),
    block shape should <= input tensor shape
    :param check_shape: check shape while run block2channel_2d(...) or block2channel_3d(...)
    :return: [h//block_H, w//block_w, channel*block_h*block_w] or
    [batch, h//block_H, w//block_w, channel*block_h*block_w]
    """
    # [h, w, channel] ==>> [h//block_H, w//block_w, channel*block_h*block_w]
    out = block2channel_3d(tf_tensor_3d, block_shape, False, check_shape)
    return tf.signal.dct(tf.cast(out, dtype=tf.float32))
