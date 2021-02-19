import numpy as np
from scipy import fft


def block2channel_2d(np_array_2d, block_shape, channel_first=True, check_shape=True):
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
    (Waring: Do not support batch, like shape[batch, h, w])
    :param block_shape: block shape should < & = input array shape
    :param channel_first: channel first ==>> True & channel last ==>> False;
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

    if channel_first:
        t2 = np.transpose(t1, (1, 3, 0, 2))
        out = np.reshape(t2, (block_h * block_w, array_h // block_h, array_w // block_w))
    else:
        t2 = np.transpose(t1, (0, 2, 1, 3))
        out = np.reshape(t2, (array_h // block_h, array_w // block_w, block_h * block_w))

    return out


def dct2channel(np_array_2d, block_shape, channel_first=True, check_shape=True):
    out = block2channel_2d(np_array_2d, block_shape, channel_first, check_shape)
    return fft.dct(out)
