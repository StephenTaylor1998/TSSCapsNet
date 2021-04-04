import math
import tensorflow as tf
from tensorflow import Variable, sqrt, float32
from tensorflow.keras import layers, regularizers, activations


class FFC(layers.Layer):
    """
      "Folding Full Connection": A Linear like operator, support tensor shape [batch, N_dim, D],
    the operator will traverse dimension N, and perform a calculation on dimension D.
    If change params(kernel_size, strides, as_matrix),the operator will be much more complex.
    When using default parameter，the computational complexity of this operation is the same as
    matmul but with more weight-parameter.
    Example in each vector at dim D:
        X:  1  2  3 | 4  5  6 | 7  8  9
               @         @         @
        W:  1  1  1   2  2  2   1  1  1
            1  1  1   2  2  2   1  1  1
            1  1  1   2  2  2   1  1  1
               =         =         =
        Y:  6  6  6 | 30 30 30| 24 24 24
    Different With Other Operators:
        Linear:
            X:  1 2 3 4 5 6
                     @
            W:  1 1 1 1 1 1
                1 1 1 1 1 1
                1 1 1 1 1 1
                1 1 1 1 1 1
                1 1 1 1 1 1
                1 1 1 1 1 1
                     =
            Y:  1 2 3 4 5 6
        Matmul：              Flatten X and Y:
            X:  1  2  3        X:  1  2  3 | 4  5  6 | 7  8  9
                4  5  6
                7  8  9               @         @         @
                   @
            W:  1  1  1            1  1  1 | 1  1  1 | 1  1  1
                1  1  1            1  1  1 | 1  1  1 | 1  1  1
                1  1  1            1  1  1 | 1  1  1 | 1  1  1
                   =                            =
            Y:  6  6  6        X:  6  6  6 | 15 15 15| 24 24 24
               15 15 15
               24 24 24
    """

    def __init__(self, out_length, groups=1, as_matrix=False, kernel_initializer='glorot_uniform', **kwargs):
        """
        "Folding Full Connection"
        input: [batch, N_dim, D] ==>> output: [batch, N_dim, out_length * group]
        :param out_length: length of each vector after folding and linear operator.
        :param group: folding times.
        :param as_matrix: if as matrix, group and out_length will be sqrt(D) automatically.
        :return tensor shape == [batch, N_dim, out_length * group]
        """
        super(FFC, self).__init__(**kwargs)
        self.out_length = out_length
        self.groups = groups
        self.as_matrix = as_matrix
        self.kernel_initializer = kernel_initializer
        self.folding = None
        self.reverse_folding = None
        self.W = None

    def build(self, input_shape):
        if self.as_matrix:
            self.groups = int(math.sqrt(input_shape[-1]))
            assert self.groups * self.groups == input_shape[-1], \
                "[ERROR] If using as_matrix==True, input_shape[-1] should = groups * groups."

        assert self.groups <= self.out_length, "[ERROR] self.group should <= self.out_numbers"

        vector_length = input_shape[-1]
        n_shape = list(input_shape[1:-1])
        group_length = vector_length // self.groups
        assert group_length * self.groups == vector_length, "vector_length must be divisible by group"
        self.kernel_initializer = tf.keras.initializers.get(self.kernel_initializer)
        self.W = self.add_weight(shape=[self.groups, group_length, self.out_length],
                                 initializer=self.kernel_initializer, name='W')
        # self.folding = layers.Reshape((-1, num_vector, self.groups, group_length))
        # self.reverse_folding = layers.Reshape((-1, num_vector, self.groups*self.out_length))
        self.folding = layers.Reshape((-1, *n_shape, self.groups, group_length))
        self.reverse_folding = layers.Reshape((-1, *n_shape, self.groups * self.out_length))

    def call(self, inputs, **kwargs):
        x = self.folding(inputs)
        x = tf.einsum('...gl,glo->...go', x, self.W)
        out = self.reverse_folding(x)
        return out


class Condense(layers.Layer):
    """
      A conv1d operator, support tensor shape [batch, N, D],
    the operator will traverse dimension N, and perform a calculation on [batch, n:n+rate, :].
    It is recommended to use rate == strings to avoid redundant computation.
    Example:
        parameter(input_shape=[1, 4, 4], out_length=2, rate=2, strides=2)
        X:  1  1  1  1      W:                              Y:
            1  1  1  1          1  1  1  1 | 2  2  2  2         12 | 12
            2  2  2  2          2  2  2  2 | 1  1  1  1         24 | 24
            2  2  2  2
    Output:
        [batch, N/rate, out_length]
    """

    def __init__(self, out_length, rate=2, strides=2, **kwargs):
        super(Condense, self).__init__(**kwargs)
        self.out_length = out_length
        self.rate = rate
        self.sparse_extraction = layers.Conv1D(out_length, rate, strides=strides, use_bias=False,
                                               kernel_regularizer=regularizers.L2(5e-5))
        self.params = Variable(self.out_length * self.rate, dtype=float32, trainable=False, name='params')

        self.activation = layers.ELU()

    def call(self, inputs, **kwargs):
        out = self.sparse_extraction(inputs)
        out = out / sqrt(self.params)
        out = self.activation(out)

        return out


class CapsFPN(layers.Layer):
    def __init__(self, out_length, rate=None, strides=None):
        super(CapsFPN, self).__init__(name='caps_fpn')
        strides = [2, 2, 2, 1] if strides is None else strides
        rate = [2, 2, 2, 1] if rate is None else rate
        self.out_length = out_length
        self.condense1 = Condense(self.out_length, rate[0], strides[0])
        self.layer_normal1 = layers.LayerNormalization()
        self.condense2 = Condense(self.out_length, rate[1], strides[1])
        self.layer_normal2 = layers.LayerNormalization()
        self.condense3 = Condense(self.out_length, rate[2], strides[2])
        self.layer_normal3 = layers.LayerNormalization()
        self.condense4 = Condense(self.out_length, rate[3], strides[3])
        self.layer_normal4 = layers.LayerNormalization()
        self.feature_pyramid = layers.Concatenate(axis=1)
        self.layer_normal5 = layers.LayerNormalization()

    def call(self, inputs, **kwargs):
        l1 = self.condense1(inputs)
        l1 = self.layer_normal1(l1)
        l2 = self.condense2(l1)
        l2 = self.layer_normal2(l2)
        l3 = self.condense3(l2)
        l3 = self.layer_normal3(l3)
        l4 = self.condense3(l3)
        l4 = self.layer_normal4(l4)
        feature_pyramid = self.feature_pyramid([l1, l2, l3, l4])
        feature_pyramid = self.layer_normal5(feature_pyramid)
        return feature_pyramid


class Q(layers.Layer):

    def __init__(self):
        super(Q, self).__init__(name='q')
        self.ffc1 = FFC(out_length=4, group=2, as_matrix=False)
        self.layer_normal1 = layers.LayerNormalization()
        self.ffc2 = FFC(out_length=1, group=1, as_matrix=False)
        self.layer_normal2 = layers.LayerNormalization()
        self.activation = layers.ELU()

    def call(self, inputs, **kwargs):
        out = self.ffc1(inputs)
        out = self.layer_normal1(out)
        out = self.activation(out)
        out = self.ffc2(out)
        out = self.layer_normal2(out)
        out = self.activation(out)
        return out


class K(layers.Layer):

    def __init__(self, **kwargs):
        super(K, self).__init__(**kwargs)
        self.get_center = tf.reduce_sum
        self.layer_normal2 = layers.LayerNormalization()
        self.dot = layers.Dot((2, 2), normalize=True)

    def call(self, inputs, **kwargs):
        global_center = self.get_center(inputs, axis=1, keepdims=True)
        global_center = self.layer_normal2(global_center)
        out = self.dot([inputs, global_center])
        return out


class CapsuleMapping(layers.Layer):

    def __init__(self):
        super(CapsuleMapping, self).__init__(name='capsule_mapping')
        self.mapping = layers.Dot((2, 2), normalize=True)
        self.attention = layers.Dot((2, 1), normalize=False)

    def call(self, inputs, **kwargs):
        k, q, v = inputs
        mapping = self.mapping([k, q])
        out = self.attention([mapping, v])
        return out


class Routing(layers.Layer):
    def __init__(self):
        super(Routing, self).__init__(name='routing')
        self.fpn = CapsFPN(out_length=8)
        self.q = Q()
        self.k = K()
        self.mapping = CapsuleMapping()

    def call(self, inputs, **kwargs):
        feature_pyramid, global_caps = self.fpn(inputs)
        q = self.q(feature_pyramid)
        k = self.k((feature_pyramid, global_caps))
        out = self.mapping((k, q, feature_pyramid))

        return out
