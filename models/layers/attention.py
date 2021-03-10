import tensorflow as tf
from tensorflow.keras import layers


class CapsuleAttentionBlock(tf.keras.layers.Layer):

    def __init__(self, ):
        super(CapsuleAttentionBlock, self).__init__()
        self.pool = None
        self.squeeze = None
        self.relu = layers.ReLU()
        self.excitation = None
        self.sigmoid = tf.sigmoid

    def build(self, input_shape):
        self.batch_size = input_shape[0]
        # self.in_channel = input_shape[-1]
        self.built = True

    def call(self, inputs, **kwargs):
        return
