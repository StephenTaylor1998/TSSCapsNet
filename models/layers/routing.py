import tensorflow as tf
from .attention import CapsuleAttentionBlock

def squash(s):
    """
    Squash activation function presented in 'Dynamic routinig between capsules'.
    ...

    Parameters
    ----------
    s: tensor
        input tensor
    """
    n = tf.norm(s, axis=-1, keepdims=True)
    return tf.multiply(n ** 2 / (1 + n ** 2) / (n + tf.keras.backend.epsilon()), s)


class AttentionDigitCaps(tf.keras.layers.Layer):
    """
    Create a light attention digit caps layer.

    ...

    Attributes
    ----------
    C: int
        number of Digit capsules
    L: int
        Digit capsules dimension (number of properties)
    routing: int
        number of routing iterations
    kernel_initializer:
        matrix W kernel initializer

    Methods
    -------
    call(inputs)
        compute the primary capsule layer
    """

    def __init__(self, C, L, routing=None, kernel_initializer='glorot_uniform', **kwargs):
        super(AttentionDigitCaps, self).__init__(**kwargs)
        self.C = C
        self.L = L
        self.routing = routing
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.attention = None

    def build(self, input_shape):
        assert len(input_shape) >= 5, "The input Tensor should have shape=[None,H,W,input_C,input_L]"
        H = input_shape[-4]
        W = input_shape[-3]
        input_C = input_shape[-2]
        input_L = input_shape[-1]

        self.W = self.add_weight(shape=[H * W * input_C, input_L, self.L * self.C], initializer=self.kernel_initializer,
                                 name='W')
        self.biases = self.add_weight(shape=[self.C, self.L], initializer='zeros', name='biases')
        self.attention = CapsuleAttentionBlock()
        self.built = True

    def call(self, inputs, **kwargs):
        H, W, input_C, input_L = inputs.shape[1:]  # input shape=(None,H,W,input_C,input_L)
        x = tf.reshape(inputs, (-1, H * W * input_C, input_L))  # x shape=(None,H*W*input_C,input_L)

        u = tf.einsum('...ji,jik->...jk', x, self.W)  # u shape=(None,H*W*input_C,C*L)
        u = tf.reshape(u, (-1, H * W * input_C, self.C, self.L))  # u shape=(None,H*W*input_C,C,L)

        if self.routing == 'attention':
            s = tf.reduce_sum(u, axis=1, keepdims=True)
            s += self.biases
            s = self.attention(s[:, 0, ...])
            s = tf.expand_dims(s, 1)
            v = squash(s)
            v = v[:, 0, ...]
        elif self.routing:
            # Hinton's routing
            b = tf.zeros(tf.shape(u)[:-1])[..., None]  # b shape=(None,H*W*input_C,C,1) -> (None,i,j,1)
            for r in range(self.routing):
                c = tf.nn.softmax(b, axis=2)  # c shape=(None,H*W*input_C,C,1) -> (None,i,j,1)
                s = tf.reduce_sum(tf.multiply(u, c), axis=1, keepdims=True)  # s shape=(None,1,C,L)
                s += self.biases
                s = self.attention(s[:, 0, ...])
                s = tf.expand_dims(s, 1)
                v = squash(s)  # v shape=(None,1,C,L)
                if r < self.routing - 1:
                    b += tf.reduce_sum(tf.multiply(u, v), axis=-1, keepdims=True)
            v = v[:, 0, ...]  # v shape=(None,C,L)
        else:
            s = tf.reduce_sum(u, axis=1, keepdims=True)
            s += self.biases
            v = squash(s)
            v = v[:, 0, ...]
        return v

    def compute_output_shape(self, input_shape):
        return None, self.C, self.L

    def get_config(self):
        config = {
            'C': self.C,
            'L': self.L,
            'routing': self.routing
        }
        base_config = super(AttentionDigitCaps, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
