import numpy as np
# from .dct2channel import DCT_Block2Channel
from DCT.dct2channel import DCT_Block2Channel


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

    channel_first = DCT_Block2Channel(array_9x9_block_3x3, block_shape=(3, 3), channel_first=True)
    print(channel_first[0, :, :])

    channel_last = DCT_Block2Channel(array_9x9_block_3x3, block_shape=(3, 3), channel_first=False)
    print(channel_last[:, :, 0])

    print(array_16x12_block_4x4)

    channel_first = DCT_Block2Channel(array_16x12_block_4x4, block_shape=(5, 4), channel_first=True)
    print(channel_first[0, :, :])

    channel_last = DCT_Block2Channel(array_16x12_block_4x4, block_shape=(4, 4), channel_first=False)
    print(channel_last[:, :, 0])


if __name__ == '__main__':
    dct2channel_test()
