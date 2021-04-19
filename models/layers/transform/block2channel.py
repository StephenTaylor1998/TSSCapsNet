# Copyright 2021 Hang-Chi Shen. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
import tensorflow as tf


def block2channel_2d(np_array_2d, block_shape, output_channel_first=False, check_shape=True):
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
    :param np_array_2d: input numpy array([h, w])
    (Waring: Do not support batch, like shape[batch, h, w])
    :param block_shape: (block_h, block_w) example (2, 2),
    block shape should <= input tensor shape
    :param output_channel_first: channel first ==>> True & channel last ==>> False;
    :param check_shape: check shape before operator
    :return: 3d numpy array
    """
    array_h = np_array_2d.shape[-2]
    array_w = np_array_2d.shape[-1]

    block_h = block_shape[0]
    block_w = block_shape[-1]

    if check_shape:
        assert len(np_array_2d.shape) == 2, \
            "check the dims of np_array_2d, the func recommend len(np_array_2d.shape) == 2"
        assert array_h % block_h == 0, \
            "check the array_h and block_h, the func recommend array_h % block_h == 0"
        assert array_w % block_w == 0, \
            "check the array_w and block_w, the func recommend array_w % block_w == 0"

    t1 = np.reshape(np_array_2d, (array_h // block_h, block_h, array_w // block_w, block_w))

    if output_channel_first:
        t2 = np.transpose(t1, (1, 3, 0, 2))
        out = np.reshape(t2, (block_h * block_w, array_h // block_h, array_w // block_w))
    else:
        t2 = np.transpose(t1, (0, 2, 1, 3))
        out = np.reshape(t2, (array_h // block_h, array_w // block_w, block_h * block_w))

    return out


def block2channel_3d(np_array_3d, block_shape, output_channel_first=False, check_shape=True):
    """
    example:
    block_shape=(2, 2)
    channel_first=False
    np_array_3d=
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
    :param np_array_3d: input numpy array([h, w, channel])
    (Waring: Do not support batch, like shape[batch, h, w, channel])
    :param block_shape: (block_h, block_w) example (2, 2),
    block shape should <= input tensor shape
    :param output_channel_first: channel first ==>> True & channel last ==>> False;
    :param check_shape: check shape before operator
    :return: 3d numpy array
    """
    array_h = np_array_3d.shape[-3]
    array_w = np_array_3d.shape[-2]
    array_c = np_array_3d.shape[-1]

    block_h = block_shape[0]
    block_w = block_shape[-1]

    if check_shape:
        assert len(np_array_3d.shape) == 3, \
            "check the dims of np_array_2d, the func recommend len(np_array_3d.shape) == 3"
        assert array_h % block_h == 0, \
            "check the array_h and block_h, the func recommend array_h % block_h == 0"
        assert array_w % block_w == 0, \
            "check the array_w and block_w, the func recommend array_w % block_w == 0"

    t1 = np.reshape(np_array_3d, (array_h // block_h, block_h, array_w // block_w, block_w, array_c))

    if output_channel_first:
        t2 = np.transpose(t1, (4, 1, 3, 0, 2))
        out = np.reshape(t2, (array_c * block_h * block_w, array_h // block_h, array_w // block_w))
    else:
        t2 = np.transpose(t1, (0, 2, 4, 1, 3))
        out = np.reshape(t2, (array_h // block_h, array_w // block_w, array_c * block_h * block_w))

    return out


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
