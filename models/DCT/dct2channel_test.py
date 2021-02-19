import numpy as np

from models.DCT.dct2channel import block2channel_2d
from models.DCT.dct2channel_np import block2channel_2d as block2channel_2d_np


def dct2channel_test():
    array_9x9_block_3x3 = np.array([
        [1, 2, 3,  1, 2, 3,  1, 2, 3],
        [4, 5, 6,  4, 5, 6,  4, 5, 6],
        [7, 8, 9,  7, 8, 9,  7, 8, 9],

        [1, 2, 3,  1, 2, 3,  1, 2, 3],
        [4, 5, 6,  4, 5, 6,  4, 5, 6],
        [7, 8, 9,  7, 8, 9,  7, 8, 9],

        [1, 2, 3,  1, 2, 3,  1, 2, 3],
        [4, 5, 6,  4, 5, 6,  4, 5, 6],
        [7, 8, 9,  7, 8, 9,  7, 8, 9],
    ])

    array_16x12_block_4x4 = np.array([

        [1,  2,  3,  4,   1,  2,  3,  4,   1,  2,  3,  4],
        [5,  6,  7,  8,   5,  6,  7,  8,   5,  6,  7,  8],
        [9,  10, 11, 12,  9,  10, 11, 12,  9,  10, 11, 12],
        [13, 14, 15, 16,  13, 14, 15, 16,  13, 14, 15, 16],

        [1,  2,  3,  4,   1,  2,  3,  4,   1,  2,  3,  4],
        [5,  6,  7,  8,   5,  6,  7,  8,   5,  6,  7,  8],
        [9,  10, 11, 12,  9,  10, 11, 12,  9,  10, 11, 12],
        [13, 14, 15, 16,  13, 14, 15, 16,  13, 14, 15, 16],

        [1,  2,  3,  4,   1,  2,  3,  4,   1,  2,  3,  4],
        [5,  6,  7,  8,   5,  6,  7,  8,   5,  6,  7,  8],
        [9,  10, 11, 12,  9,  10, 11, 12,  9,  10, 11, 12],
        [13, 14, 15, 16,  13, 14, 15, 16,  13, 14, 15, 16],

        [1,  2,  3,  4,   1,  2,  3,  4,   1,  2,  3,  4],
        [5,  6,  7,  8,   5,  6,  7,  8,   5,  6,  7,  8],
        [9,  10, 11, 12,  9,  10, 11, 12,  9,  10, 11, 12],
        [13, 14, 15, 16,  13, 14, 15, 16,  13, 14, 15, 16],

    ])

    print(array_9x9_block_3x3)

    channel_first = block2channel_2d_np(array_9x9_block_3x3, block_shape=(3, 3), channel_first=True)
    print(channel_first[0, :, :])
    channel_first = block2channel_2d(array_9x9_block_3x3, block_shape=(3, 3), channel_first=True)
    print(channel_first[0, :, :])

    channel_last = block2channel_2d_np(array_9x9_block_3x3, block_shape=(3, 3), channel_first=False)
    print(channel_last[:, :, 0])
    channel_last = block2channel_2d(array_9x9_block_3x3, block_shape=(3, 3), channel_first=False)
    print(channel_last[:, :, 0])

    print(array_16x12_block_4x4)

    channel_first = block2channel_2d_np(array_16x12_block_4x4, block_shape=(4, 4), channel_first=True)
    print(channel_first[0, :, :])
    channel_first = block2channel_2d(array_16x12_block_4x4, block_shape=(4, 4), channel_first=True)
    print(channel_first[0, :, :])

    channel_last = block2channel_2d_np(array_16x12_block_4x4, block_shape=(4, 4), channel_first=False)
    print(channel_last[:, :, 0])
    channel_last = block2channel_2d(array_16x12_block_4x4, block_shape=(4, 4), channel_first=False)
    print(channel_last[:, :, 0])

    return channel_first, channel_last


if __name__ == '__main__':
    from scipy.fft import dct
    import tensorflow

    c_frist, c_last = dct2channel_test()

    out = dct(c_last)

    print(out)

    out = tensorflow.signal.dct(c_last.astype(np.float32))

    print(out[0, 0])

    print(out[0, :, 0])

    print(out[:, 0, 0])


