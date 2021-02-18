import numpy as np


def DCT_Block2Channel(np_array_2d, block_shape, channel_first=True, check_shape=True):
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
