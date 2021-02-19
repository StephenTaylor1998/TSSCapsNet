import numpy as np
# block2channel_test
from models.DCT.dct2channel_tf import Block2Channel2d, Block2Channel3d
from models.DCT.dct2channel_np import block2channel_2d as block2channel_2d_np

# dct2channel_test
from models.DCT.dct2channel_np import dct2channel as dct2channel_np
from models.DCT.dct2channel_np import block2channel_3d as block2channel_3d_np
from models.DCT.dct2channel_tf import DCTLayer3d, DCTLayer2d
from models.DCT.dct2channel_tf import Block2Channel3d as block2channel_3d_tf

# test in model
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import losses


def block2channel_test():
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

    channel_first = block2channel_2d_np(array_9x9_block_3x3, block_shape=(3, 3), output_channel_first=True)
    print(channel_first[0, :, :])
    channel_first = Block2Channel2d(array_9x9_block_3x3, block_shape=(3, 3), output_channel_first=True)
    print(channel_first[0, :, :])

    channel_last = block2channel_2d_np(array_9x9_block_3x3, block_shape=(3, 3), output_channel_first=False)
    print(channel_last[:, :, 0])
    channel_last = Block2Channel2d(array_9x9_block_3x3, block_shape=(3, 3), output_channel_first=False)
    print(channel_last[:, :, 0])

    print(array_16x12_block_4x4)

    channel_first = block2channel_2d_np(array_16x12_block_4x4, block_shape=(4, 4), output_channel_first=True)
    print(channel_first[0, :, :])
    channel_first = Block2Channel2d(array_16x12_block_4x4, block_shape=(4, 4), output_channel_first=True)
    print(channel_first[0, :, :])

    channel_last = block2channel_2d_np(array_16x12_block_4x4, block_shape=(4, 4), output_channel_first=False)
    print(channel_last[:, :, 0])
    channel_last = Block2Channel2d(array_16x12_block_4x4, block_shape=(4, 4), output_channel_first=False)
    print(channel_last[:, :, 0])


def dct2channel_test():
    array_2x4x4 = np.array([
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
    array_4x4x2 = np.transpose(array_2x4x4, (1, 2, 0))

    array_8x2x2 = block2channel_3d_np(array_4x4x2, (2, 2), output_channel_first=True)
    print(array_8x2x2)

    array_2x2x8 = block2channel_3d_np(array_4x4x2, (2, 2), output_channel_first=False)
    print(array_2x2x8)

    array_2x2x8 = dct2channel_np(array_4x4x2, (2, 2))
    print(array_2x2x8)


def test_in_model():
    array_2x4x4 = np.array([
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
    array_4x4x2 = np.transpose(array_2x4x4, (1, 2, 0))
    array_2x4x4x2 = np.concatenate([[array_4x4x2], [array_4x4x2]])
    print(array_2x4x4x2.shape)

    input_tensor = layers.Input((4, 4, 2))
    print(input_tensor)
    # output_tensor = Block2Channel3d((2, 2))(input_tensor)
    output_tensor = DCTLayer3d((2, 2))(input_tensor)
    model = Model(inputs=input_tensor, outputs=output_tensor)
    model.compile()

    out = model(array_2x4x4x2)
    print(out)

    # input_tensor = layers.Input((4, 4))
    # print(input_tensor)
    # # output_tensor = Block2Channel2d((2, 2))(input_tensor)
    # output_tensor = DCTLayer2d((2, 2))(input_tensor)
    # model = Model(inputs=input_tensor, outputs=output_tensor)
    # model.compile()
    #
    # array_2x4x4 = np.array(array_2x4x4x2[:, :, :, 0])
    # print(array_2x4x4.shape)
    # out = model(array_2x4x4)
    # print(out)

    pass


if __name__ == '__main__':
    # block2channel_test()
    # dct2channel_test()
    test_in_model()
    # import tensorflow as tf
    #
    # conv = tf.keras.layers.Conv2D(filters=1, kernel_size=3)
    # array_2x4x4 = np.array([
    #     [[1, 2,  1, 2],
    #      [3, 4,  3, 4],
    #
    #      [1, 2,  1, 2],
    #      [3, 4,  3, 4]],
    #
    #     [[5, 6,  5, 6],
    #      [7, 8,  7, 8],
    #
    #      [5, 6,  5, 6],
    #      [7, 8,  7, 8]],
    # ])
    # # convert to channel last
    # array_4x4x2 = np.transpose(array_2x4x4, (1, 2, 0))
    # print(conv(array_4x4x2))