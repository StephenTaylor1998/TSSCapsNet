import tensorflow as tf
from tensorflow.keras import layers


class CapsuleAttentionBlock(tf.keras.layers.Layer):

    def __init__(self, ):
        super(CapsuleAttentionBlock, self).__init__()

    def build(self, input_shape):
        self.built = True

    def call(self, inputs, **kwargs):
        return
