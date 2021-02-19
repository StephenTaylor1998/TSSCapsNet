import numpy as np
# block2channel_test
# from models.DCT.dct2channel_tf import block2channel_2d
# from models.DCT.dct2channel_np import block2channel_2d as block2channel_2d_np
# dct2channel_test
# from models.DCT.dct2channel_tf import dct2channel
from models.DCT.dct2channel_np import dct2channel as dct2channel_np
from models.DCT.dct2channel_np import block2channel_3d as block2channel_3d_np


# def block2channel_test():
#     array_9x9_block_3x3 = np.array([
#         [1, 2, 3, 1, 2, 3, 1, 2, 3],
#         [4, 5, 6, 4, 5, 6, 4, 5, 6],
#         [7, 8, 9, 7, 8, 9, 7, 8, 9],
#
#         [1, 2, 3, 1, 2, 3, 1, 2, 3],
#         [4, 5, 6, 4, 5, 6, 4, 5, 6],
#         [7, 8, 9, 7, 8, 9, 7, 8, 9],
#
#         [1, 2, 3, 1, 2, 3, 1, 2, 3],
#         [4, 5, 6, 4, 5, 6, 4, 5, 6],
#         [7, 8, 9, 7, 8, 9, 7, 8, 9],
#     ])
#
#     array_16x12_block_4x4 = np.array([
#
#         [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
#         [5, 6, 7, 8, 5, 6, 7, 8, 5, 6, 7, 8],
#         [9, 10, 11, 12, 9, 10, 11, 12, 9, 10, 11, 12],
#         [13, 14, 15, 16, 13, 14, 15, 16, 13, 14, 15, 16],
#
#         [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
#         [5, 6, 7, 8, 5, 6, 7, 8, 5, 6, 7, 8],
#         [9, 10, 11, 12, 9, 10, 11, 12, 9, 10, 11, 12],
#         [13, 14, 15, 16, 13, 14, 15, 16, 13, 14, 15, 16],
#
#         [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
#         [5, 6, 7, 8, 5, 6, 7, 8, 5, 6, 7, 8],
#         [9, 10, 11, 12, 9, 10, 11, 12, 9, 10, 11, 12],
#         [13, 14, 15, 16, 13, 14, 15, 16, 13, 14, 15, 16],
#
#         [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
#         [5, 6, 7, 8, 5, 6, 7, 8, 5, 6, 7, 8],
#         [9, 10, 11, 12, 9, 10, 11, 12, 9, 10, 11, 12],
#         [13, 14, 15, 16, 13, 14, 15, 16, 13, 14, 15, 16],
#
#     ])
#
#     print(array_9x9_block_3x3)
#
#     channel_first = block2channel_2d_np(array_9x9_block_3x3, block_shape=(3, 3), output_channel_first=True)
#     print(channel_first[0, :, :])
#     channel_first = block2channel_2d(array_9x9_block_3x3, block_shape=(3, 3), output_channel_first=True)
#     print(channel_first[0, :, :])
#
#     channel_last = block2channel_2d_np(array_9x9_block_3x3, block_shape=(3, 3), output_channel_first=False)
#     print(channel_last[:, :, 0])
#     channel_last = block2channel_2d(array_9x9_block_3x3, block_shape=(3, 3), output_channel_first=False)
#     print(channel_last[:, :, 0])
#
#     print(array_16x12_block_4x4)
#
#     channel_first = block2channel_2d_np(array_16x12_block_4x4, block_shape=(4, 4), output_channel_first=True)
#     print(channel_first[0, :, :])
#     channel_first = block2channel_2d(array_16x12_block_4x4, block_shape=(4, 4), output_channel_first=True)
#     print(channel_first[0, :, :])
#
#     channel_last = block2channel_2d_np(array_16x12_block_4x4, block_shape=(4, 4), output_channel_first=False)
#     print(channel_last[:, :, 0])
#     channel_last = block2channel_2d(array_16x12_block_4x4, block_shape=(4, 4), output_channel_first=False)
#     print(channel_last[:, :, 0])


def dct2channel_test():
    array_2x4x4_block_2x2 = np.array([
        [[1, 2,  1, 2],
         [3, 4,  3, 4],

         [1, 2,  1, 2],
         [3, 4,  3, 4]],

        [[5, 6,  5, 6],
         [7, 8,  7, 8],

         [5, 6,  5, 6],
         [7, 8,  7, 8]],
    ])
    # convert to channel last
    array_4x4x2_block_2x2 = np.transpose(array_2x4x4_block_2x2, (1, 2, 0))

    array_8x2x2 = block2channel_3d_np(array_4x4x2_block_2x2, (2, 2), output_channel_first=True)
    print(array_8x2x2)

    array_2x2x8 = block2channel_3d_np(array_4x4x2_block_2x2, (2, 2), output_channel_first=False)
    print(array_2x2x8)

    array_2x2x8 = dct2channel_np(array_4x4x2_block_2x2, (2, 2))
    print(array_2x2x8)

    pass


if __name__ == '__main__':
    # block2channel_test()
    dct2channel_test()
