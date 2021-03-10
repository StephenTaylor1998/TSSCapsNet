import tensorflow as tf
from tensorflow.keras import layers


class CapsuleAttentionBlock(tf.keras.layers.Layer):
    """
    Attention-block(squeeze & excitation) for capsule-format data.
    example:
        input_tensor ==>> [batch, caps_number, caps_length]
        squeeze_tensor ==>> [batch, caps_number//2]
        excitation_tensor ==>> [batch, caps_number]
        after_reshape_tensor==>> [batch, caps_number, 1]
        final_output_tensor ==>> [batch, caps_number, caps_length]

    Operator:
        final_output_tensor = input_tensor * after_reshape_tensor
    """
    def __init__(self):
        super(CapsuleAttentionBlock, self).__init__()
        self.pool = layers.GlobalAveragePooling1D(data_format='channels_first')
        self.squeeze = None
        self.relu = layers.ReLU()
        self.excitation = None
        self.reshape = None
        self.sigmoid = tf.sigmoid

    def build(self, input_shape):
        self.squeeze = layers.Dense(input_shape[1] // 2)
        self.excitation = layers.Dense(input_shape[1])
        self.reshape = tf.keras.layers.Reshape((input_shape[1], 1))
        self.built = True

    def call(self, inputs, **kwargs):
        x = self.pool(inputs)
        x = self.squeeze(x)
        x = self.excitation(x)
        x = self.reshape(x)
        x = self.sigmoid(x)
        return inputs * x
