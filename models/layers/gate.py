import tensorflow as tf
from tensorflow.keras import layers
from .gumbel import GumbelSoftmax


class GateModule(tf.keras.layers.Layer):

    def __init__(self, act='relu'):
        super(GateModule, self).__init__()
        self.batch_size = None
        self.in_channel = None
        self.avg_pool = layers.GlobalAveragePooling2D()
        self.reshape = None
        self.inp_gate = None
        self.inp_gate_l = None
        self.channel_reshape = None
        self.gumbel_softmax = GumbelSoftmax()

        if act == 'relu':
            self.relu = layers.ReLU()
        elif act == 'relu6':
            self.relu = layers.ReLU(max_value=6.)
        else:
            self.relu = None
            raise NotImplementedError

    def build(self, input_shape):
        self.batch_size = input_shape[-4]
        self.in_channel = input_shape[-1]
        self.reshape = layers.Reshape((1, 1, self.in_channel))
        self.inp_gate = tf.keras.Sequential([
            layers.Conv2D(self.in_channel, kernel_size=1, strides=1, use_bias=True),
            layers.BatchNormalization(),
            self.relu,
        ])
        self.inp_gate_l = layers.Conv2D(self.in_channel * 2, kernel_size=1, strides=1, groups=self.in_channel,
                                        use_bias=True)
        self.channel_reshape = layers.Reshape((self.in_channel, 2))
        self.built = True

    def call(self, inputs, temperature=1., **kwargs):
        # (batch, h, w, channel) ==>> (batch, channel)
        hatten = self.avg_pool(inputs)
        # (batch, channel) ==>> (batch, 1, 1, channel)
        hatten = self.reshape(hatten)
        # (batch, 1, 1, channel) ==>> (batch, 1, 1, channel)
        hatten_d = self.inp_gate(hatten)
        # (batch, 1, 1, channel) ==>> (batch, 1, 1, channel*2)
        hatten_d = self.inp_gate_l(hatten_d)
        # (batch, 1, 1, channel) ==>> (batch, channel, 2)
        hatten_d = self.channel_reshape(hatten_d)
        # (batch, channel, 2) ==>> (batch, channel)
        hatten_d = self.gumbel_softmax(hatten_d, temp=temperature, force_hard=True)
        # (batch, channel) ==>> (batch, 1, 1, channel)
        hatten_d = self.reshape(hatten_d)
        # (batch, h, w, channel) * (batch, 1, 1, channel) ==>> (batch, h, w, channel)
        x = inputs * hatten_d
        return x, hatten_d
