import tensorflow as tf


class Block2Channel2d(tf.keras.layers.Layer):
    """
        example:
        block_shape=(2, 2)
        output_channel_first=False
        input=
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
        :param block_shape: (block_h, block_w) example (2, 2),
        block shape should <= input tensor shape
        :param output_channel_first: channel first ==>> True & channel last ==>> False;
        :param check_shape: check shape before operator
        :return: [batch, h//block_H, w//block_w, block_h*block_w]
        """
    def __init__(self, block_shape, output_channel_first=False, check_shape=True, **kwargs):
        super(Block2Channel2d, self).__init__(**kwargs)

        self.batch_size = None
        self.tensor_h = None
        self.tensor_w = None

        self.block_h = block_shape[0]
        self.block_w = block_shape[-1]

        self.output_channel_first = output_channel_first
        self.check_shape = check_shape

    def build(self, input_shape):
        print(input_shape)
        self.batch_size = input_shape[-3]
        self.tensor_h = input_shape[-2]
        self.tensor_w = input_shape[-1]

        if self.check_shape:
            assert self.tensor_h % self.block_h == 0, \
                "the tensor_h{} and block_h{}, recommend tensor_h % block_h == 0".format(self.tensor_h, self.block_h)
            assert self.tensor_w % self.block_w == 0, \
                "the tensor_w{} and block_w{}, recommend tensor_w % block_w == 0".format(self.tensor_w, self.block_w)

    def call(self, inputs, **kwargs):

        t1 = tf.keras.layers.Reshape((self.tensor_h // self.block_h,
                                      self.block_h,
                                      self.tensor_w // self.block_w,
                                      self.block_w))(inputs)

        if self.output_channel_first:
            t2 = tf.transpose(t1, (0, 2, 4, 1, 3))
            out = tf.keras.layers.Reshape((self.block_h * self.block_w,
                                           self.tensor_h // self.block_h,
                                           self.tensor_w // self.block_w))(t2)

        else:
            t2 = tf.transpose(t1, (0, 1, 3, 2, 4))
            out = tf.keras.layers.Reshape((self.tensor_h // self.block_h,
                                           self.tensor_w // self.block_w,
                                           self.block_h * self.block_w))(t2)


        return out


class Block2Channel3d(tf.keras.layers.Layer):
    """
    example:
    block_shape=(2, 2)
    output_channel_first=False
    input=
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
    :param block_shape: (block_h, block_w) example (2, 2),
    block shape should <= input tensor shape
    :param output_channel_first: channel first ==>> True & channel last ==>> False;
    :param check_shape: check shape before operator
    :return: [batch, h//block_H, w//block_w, channel*block_h*block_w]
    """
    def __init__(self, block_shape, output_channel_first=False, check_shape=True, **kwargs):
        super(Block2Channel3d, self).__init__(**kwargs)
        self.batch_size = None
        self.tensor_h = None
        self.tensor_w = None
        self.tensor_c = None
        self.block_h = block_shape[0]
        self.block_w = block_shape[-1]

        self.output_channel_first = output_channel_first
        self.check_shape = check_shape

    def build(self, input_shape):
        self.batch_size = input_shape[-4]
        self.tensor_h = input_shape[-3]
        self.tensor_w = input_shape[-2]
        self.tensor_c = input_shape[-1]

        if self.check_shape:
            assert len(input_shape) >= 4, \
                "dims of tf_tensor_3d is {}, recommend len(tf_tensor_3d.shape) >= 4".format(input_shape)
            assert self.tensor_h % self.block_h == 0, \
                "the tensor_h{} and block_h{}, recommend tensor_h % block_h == 0".format(self.tensor_h, self.block_h)
            assert self.tensor_w % self.block_w == 0, \
                "the tensor_w{} and block_w{}, recommend tensor_w % block_w == 0".format(self.tensor_w, self.block_w)

    def call(self, inputs, **kwargs):
        t1 = tf.keras.layers.Reshape((self.tensor_h // self.block_h,
                                      self.block_h,
                                      self.tensor_w // self.block_w,
                                      self.block_w,
                                      self.tensor_c))(inputs)

        if self.output_channel_first:
            t2 = tf.transpose(t1, (0, 1, 3, 5, 2, 4))
            out = tf.keras.layers.Reshape((self.tensor_c * self.block_h * self.block_w,
                                           self.tensor_h // self.block_h,
                                           self.tensor_w // self.block_w))(t2)
        else:
            t2 = tf.transpose(t1, (0, 2, 4, 1, 3, 5))
            out = tf.keras.layers.Reshape((self.tensor_h // self.block_h,
                                           self.tensor_w // self.block_w,
                                           self.tensor_c * self.block_h * self.block_w))(t2)

        return out


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
    :param check_shape: check shape while run block2channel_3d(...)
    :return: [batch, h//block_H, w//block_w, channel*block_h*block_w]
    """
    def __init__(self, block_shape, check_shape=True, **kwargs):
        super(DCTLayer3d, self).__init__(**kwargs)
        self.block_shape = block_shape
        self.check_shape = check_shape
        self.block2Channel3d = None

    def build(self, input_shape):
        self.block2Channel3d = Block2Channel3d(self.block_shape, False, self.check_shape)

    def call(self, inputs, **kwargs):
        # [batch, h, w, channel] ==>> [batch, h//block_H, w//block_w, channel*block_h*block_w]
        out = self.block2Channel3d(inputs)
        return tf.signal.dct(tf.cast(out, dtype=tf.float32))

