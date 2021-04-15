import math
import tensorflow as tf
from tensorflow.keras import layers, regularizers


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
        self.folding = None
        self.reverse_folding = None
        self.W = None
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)

    def build(self, input_shape):
        if self.as_matrix:
            self.groups = int(math.sqrt(input_shape[-1]))
            assert self.groups * self.groups == input_shape[-1], \
                "[ERROR] If using as_matrix==True, input_shape[-1] should = groups * groups."

        assert self.groups <= self.out_length, "[ERROR] self.group should <= self.out_numbers"

        vector_length = input_shape[-1]
        # n_shape = list(input_shape[1:-1])
        n_shape = input_shape[1:-1]
        group_length = vector_length // self.groups
        assert group_length * self.groups == vector_length, "vector_length must be divisible by group"
        self.W = self.add_weight(shape=[self.groups, group_length, self.out_length],
                                 initializer=self.kernel_initializer, name='W')
        # self.folding = layers.Reshape((-1, num_vector, self.groups, group_length))
        # self.reverse_folding = layers.Reshape((-1, num_vector, self.groups*self.out_length))
        self.folding = layers.Reshape((*n_shape, self.groups, group_length))
        self.reverse_folding = layers.Reshape((*n_shape, self.groups * self.out_length))
        
    def get_config(self):
        return super(FFC, self).get_config()

    def call(self, inputs, **kwargs):
        x = self.folding(inputs)
        # x = tf.einsum('...gl,glo->...go', x, self.W)
        x = tf.einsum('...ij,ijk->...ik', x, self.W)
        out = self.reverse_folding(x)
        return out


class CapsuleMappingTiny(layers.Layer):

    def __init__(self, **kwargs):
        super(CapsuleMappingTiny, self).__init__(**kwargs)
        self.mapping = layers.Dot((2, 2), normalize=True)
        self.norm1 = layers.LayerNormalization()
        self.attention = layers.Dot((2, 1), normalize=False)
        self.norm2 = layers.LayerNormalization()
        
    def get_config(self):
        return super(CapsuleMappingTiny, self).get_config()

    def call(self, inputs, **kwargs):
        k, q, v = inputs
        mapping = self.mapping([k, q])
        mapping = self.norm1(mapping)
        out = self.attention([mapping, v])
        out = self.norm2(out)
        return out


class CapsuleMapping(layers.Layer):
    def __init__(self, num_caps, caps_length=None, kernel_initializer='glorot_uniform'):
        super(CapsuleMapping, self).__init__()
        self.num_caps = num_caps
        self.caps_length = caps_length
        self.W = None
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)

    def build(self, input_shape):
        i, k = input_shape[-2], input_shape[-1]
        j, l = self.num_caps, self.caps_length
        if self.caps_length:
            weight_shape = [i, j, k, l]
        else:
            weight_shape = [i, j, k]
        self.W = self.add_weight(shape=weight_shape, initializer=self.kernel_initializer, name='W',
                                 regularizer=regularizers.L2(1e-4))

    def get_config(self):
        super(CapsuleMapping, self).get_config()

    def call(self, inputs, **kwargs):
        if self.caps_length:
            out = tf.einsum("...ik,ijkl->...jl", inputs, self.W)
        else:
            out = tf.einsum("...ik,ijk->...jk", inputs, self.W)
        return out


class CondenseTiny(layers.Layer):
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

    def __init__(self, out_length, rate=2, strides=2, regularize=1e-5, **kwargs):
        super(CondenseTiny, self).__init__(**kwargs)
        self.sparse_extraction = layers.Conv1D(out_length, rate, strides=strides, use_bias=False,
                                               kernel_regularizer=regularizers.L2(regularize))
        self.normal = layers.LayerNormalization()
        self.activation = layers.ELU()
    
    def get_config(self):
        return super(CondenseTiny, self).get_config()

    def call(self, inputs, **kwargs):
        out = self.sparse_extraction(inputs)
        out = self.normal(out)
        out = self.activation(out)
        return out


class Condense(layers.Layer):
    """
      todo: fix bugs
    """

    def __init__(self, num_caps, out_length, **kwargs):
        super(Condense, self).__init__(**kwargs)
        self.sparse_extraction = CapsuleMapping(num_caps=num_caps, caps_length=out_length)
        self.normal = layers.LayerNormalization()
        self.activation = layers.ELU()

    def get_config(self):
        super(Condense, self).get_config()

    def call(self, inputs, **kwargs):
        out = self.sparse_extraction(inputs)
        out = self.normal(out)
        out = self.activation(out)
        return out


class CapsFPNTiny(layers.Layer):
    def __init__(self, out_length, rate=None, strides=None, regularize=1e-5, **kwargs):
        super(CapsFPNTiny, self).__init__(**kwargs)
        strides = [2, 2, 2, 1] if strides is None else strides
        rate = [2, 2, 2, 1] if rate is None else rate
        self.out_length = out_length
        self.condense1 = CondenseTiny(self.out_length, rate[0], strides[0], regularize)
        self.condense2 = CondenseTiny(self.out_length, rate[1], strides[1], regularize)
        self.condense3 = CondenseTiny(self.out_length, rate[2], strides[2], regularize)
        self.condense4 = CondenseTiny(self.out_length, rate[3], strides[3], regularize)
        self.feature_pyramid = layers.Concatenate(axis=-2)

    def get_config(self):
        return super(CapsFPNTiny, self).get_config()

    def call(self, inputs, **kwargs):
        l1 = self.condense1(inputs)
        l2 = self.condense2(l1)
        l3 = self.condense3(l2)
        l4 = self.condense4(l3)
        pyramid = self.feature_pyramid([l1, l2, l3, l4])
        return pyramid


class CapsFPN(layers.Layer):
    def __init__(self, num_caps, length, **kwargs):
        super(CapsFPN, self).__init__(**kwargs)
        num_caps = [16, 8, 4, 4] if num_caps is None else num_caps
        self.condense1 = Condense(num_caps[0], length)
        self.condense2 = Condense(num_caps[1], length)
        self.condense3 = Condense(num_caps[2], length)
        self.condense4 = Condense(num_caps[3], length)
        self.feature_pyramid = layers.Concatenate(axis=-2)

    def call(self, inputs, **kwargs):
        l1 = self.condense1(inputs)
        l2 = self.condense2(l1)
        l3 = self.condense3(l2)
        l4 = self.condense4(l3)
        pyramid = self.feature_pyramid([l1, l2, l3, l4])
        return pyramid


class Q(layers.Layer):

    def __init__(self, **kwargs):
        super(Q, self).__init__(**kwargs)
        self.dim_reduction1 = FFC(out_length=2, groups=2, as_matrix=False)
        self.layer_normal1 = layers.LayerNormalization()
        self.activation1 = layers.ELU()
        self.dim_reduction2 = FFC(out_length=1, groups=1, as_matrix=False)
        self.layer_normal2 = layers.LayerNormalization()
        self.activation2 = layers.ELU()
        self.reshape = None
        self.squeeze = None
        self.relu = layers.ReLU()
        self.excitation = None
        self.reverse_reshape = None
        self.layer_normal3 = layers.LayerNormalization()

    def build(self, input_shape):
        length = input_shape[-2]
        n_shape = input_shape[1:-2]
        self.reshape = layers.Reshape((length,))
        self.squeeze = layers.Dense(length // 2)
        self.excitation = layers.Dense(length)
        self.reverse_reshape = layers.Reshape((*n_shape, length, 1))
        self.built = True

    def call(self, inputs, **kwargs):
        x = self.dim_reduction1(inputs)
        x = self.layer_normal1(x)
        x = self.activation1(x)
        x = self.dim_reduction2(x)
        x = self.layer_normal2(x)
        x = self.activation2(x)
        x = self.reshape(x)
        x = self.squeeze(x)
        x = self.relu(x)
        x = self.excitation(x)
        x = self.reverse_reshape(x)
        out = self.layer_normal3(x)
        return out


class CapsSimilarity(layers.Layer):

    def __init__(self, **kwargs):
        super(CapsSimilarity, self).__init__(**kwargs)
        self.layer_normal1 = layers.LayerNormalization()
        # self.dot = layers.Dot((2, 2), normalize=True)
        self.dot = layers.Dot((2, 2))
        self.layer_normal2 = layers.LayerNormalization()
        self.activation = layers.ELU()

    def call(self, inputs, **kwargs):
        global_center = tf.reduce_sum(inputs, axis=1, keepdims=True)
        global_center = self.layer_normal1(global_center)
        out = self.dot([inputs, global_center])
        out = self.layer_normal2(out)
        out = self.activation(out)
        return out


class Heterogeneous(layers.Layer):
    def __init__(self, num_class, **kwargs):
        super(Heterogeneous, self).__init__(**kwargs)
        self.reshape = None
        self.classify = layers.Dense(num_class)
        # self.factor = self.add_weight(shape=[2], trainable=True, regularizer=regularizers.L1L2(1e-4),
        #                               initializer=tf.keras.initializers.Ones(), name='factor')
        # self.factor = self.add_weight(shape=[2], trainable=True, name='factor')
        self.softmax = layers.Softmax()

    def build(self, input_shape):
        from_primary, caps_classify = input_shape
        num_caps, caps_length = from_primary[-2], from_primary[-1]
        self.reshape = layers.Reshape((num_caps * caps_length,))
        self.built = True

    def get_config(self):
        return super(Heterogeneous, self).get_config()

    def call(self, inputs, **kwargs):
        from_primary, caps_classify = inputs
        from_primary = self.reshape(from_primary)
        classify = self.classify(from_primary)
        # mean = tf.abs(tf.reduce_mean(self.factor))
        # mean = tf.reduce_mean(self.factor)
        # out = (self.softmax(classify) * self.factor[0] + self.softmax(caps_classify) * self.factor[1]) / mean
        out = (self.softmax(classify) * 0.5 + self.softmax(caps_classify) * 0.5)
        return out


