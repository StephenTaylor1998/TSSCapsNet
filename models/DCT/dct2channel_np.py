import numpy as np
from scipy import fft


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
    :param block_shape: block shape should < & = input array shape
    :param output_channel_first: channel first ==>> True & channel last ==>> False;
    :param check_shape: check shape before operator
    :return: 3d numpy array
    """
    array_h = np_array_2d.shape[-2]
    array_w = np_array_2d.shape[-1]

    block_h = block_shape[-2]
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
    :param block_shape: block shape should < & = input array shape
    :param output_channel_first: channel first ==>> True & channel last ==>> False;
    :param check_shape: check shape before operator
    :return: 3d numpy array
    """
    array_h = np_array_3d.shape[-3]
    array_w = np_array_3d.shape[-2]
    array_c = np_array_3d.shape[-1]

    block_h = block_shape[-2]
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


def dct2channel(np_array, block_shape, check_shape=True):
    """
    Convert 2d or 3d numpy array to 3d, then do DCT op to last dimension.
    :param np_array: 2d or 3d array
    :param block_shape: block shape should < & = input array shape
    :param check_shape: check shape while run block2channel_2d(...) or block2channel_3d(...)
    :return: 3d numpy array, the same like output of block2channel_2d(...) and block2channel_2d(...)
    """
    if len(np_array.shape) == 2:
        out = block2channel_2d(np_array, block_shape, False, check_shape)

    elif len(np_array.shape) == 3:
        out = block2channel_3d(np_array, block_shape, False, check_shape)

    else:
        print("shape {} not support, recommend [h, w] or [h, w, channel]".format(np_array.shape))
        raise NotImplementedError

    return fft.dct(out.astype(np.float32))


