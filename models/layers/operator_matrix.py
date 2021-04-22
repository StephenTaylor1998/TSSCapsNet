import tensorflow as tf
from typing import Union
from tensorflow.keras import layers, regularizers


class PartialMatrix(layers.Layer):
    """
      A partial matrix operator.
    Example:
    >>>input_tensor = tf.ones((1, 16, 4, 4), dtype=tf.float32)
    >>>layer = PartialMatrix(num_capsule=8, rate=2, matrix_shape=(4, 4))
    >>>out = layer(input_tensor)
    >>>print(out.shape)
    (1, 8, 4, 4)
    >>>input_tensor = tf.ones((1, 16, 4, 4), dtype=tf.float32)
    >>>layer = PartialMatrix(num_capsule=8, matrix_shape=(4, 4))
    >>>out = layer(input_tensor)
    >>>print(out.shape)
    (1, 8, 4, 4)
    >>>input_tensor = tf.ones((1, 16, 4, 4), dtype=tf.float32)
    >>>layer = PartialMatrix(rate=2, matrix_shape=(4, 4))
    >>>out = layer(input_tensor)
    >>>print(out.shape)
    (1, 8, 4, 4)
    """
    def __init__(self, num_capsule: int = None, rate: int = None, matrix_shape: tuple = (4, 4),
                 kernel_initializer='glorot_uniform', regularize=1e-4):
        super(PartialMatrix, self).__init__()
        if len(matrix_shape) != 2:
            raise ValueError("[ERROR]Parameter 'matrix_shape' should be a tuple with 2 element, for example (4, 4)")
        self.num_capsule = num_capsule
        self.rate = rate
        self.matrix_shape = matrix_shape
        self.kernel_initializer = kernel_initializer
        self.regularize = regularize
        self.W = None
        self.reshape = None

    def get_config(self):
        return super(PartialMatrix, self).get_config()

    def build(self, input_shape):
        num_origin_capsule = input_shape[-3]

        if self.num_capsule and num_origin_capsule % self.num_capsule == 0:
            self.rate = int(num_origin_capsule / self.num_capsule) if self.rate is None else self.rate
        elif self.rate and num_origin_capsule % self.rate == 0:
            self.num_capsule = int(num_origin_capsule / self.rate) if self.num_capsule is None else self.num_capsule
        else:
            raise ValueError("[ERROR]When check 'num_origin_capsule' and 'num_capsule', recommend select one of them.")

        if num_origin_capsule != self.num_capsule * self.rate:
            raise ValueError("[ERROR]When check 'num_origin_capsule == self.num_capsule * self.rate'.")

        self.W = self.add_weight("W", shape=[self.rate, *self.matrix_shape], dtype=tf.float32,
                                 initializer=self.kernel_initializer, regularizer=regularizers.L2(self.regularize))
        self.reshape = layers.Reshape((self.num_capsule, self.rate, *self.matrix_shape))

    def call(self, inputs, **kwargs):
        inputs = self.reshape(inputs)
        inputs = tf.einsum('...hijk,ikl->...hjl', inputs, self.W)
        return inputs


class GlobalMatrix(layers.Layer):
    """
      A global matrix operator.
    Example:
    >>>input_tensor = tf.ones((1, 16, 4, 4), dtype=tf.float32)
    >>>layer = GlobalMatrix(num_capsule=8, matrix_shape=(4, 4))
    >>>out = layer(input_tensor)
    >>>print(out.shape)
    (1, 8, 4, 4)
    """
    def __init__(self, num_capsule: int, matrix_shape=(4, 4), kernel_initializer='glorot_uniform', regularize=1e-4):
        super(GlobalMatrix, self).__init__()
        if len(matrix_shape) != 2:
            raise ValueError("[ERROR]Parameter 'matrix_shape' should be a tuple with 2 element, for example (4, 4)")

        self.W = self.add_weight("W", shape=[num_capsule, *matrix_shape], dtype=tf.float32,
                                 initializer=kernel_initializer, regularizer=regularizers.L2(regularize))

    def get_config(self):
        return super(GlobalMatrix, self).get_config()

    def call(self, inputs, **kwargs):
        inputs = tf.einsum('...ijk,hkl->...hjl', inputs, self.W)
        return inputs


class CondenseTiny(layers.Layer):
    """
      A tiny condense operator, support tensor shape [batch, N, D, D],
    the operator will traverse dimension N, and perform a calculation on [batch, n:n+rate, :, :].
    Example:
    >>>input_tensor = tf.ones((1, 16, 4, 4), dtype=tf.float32)
    >>>layer = CondenseTiny(rate=2, matrix_shape=(4, 4))
    >>>out = layer(input_tensor)
    >>>print(out.shape)
    (1, 8, 4, 4)
    """
    def __init__(self, rate=2, matrix_shape=(4, 4), regularize=1e-5, **kwargs):
        super(CondenseTiny, self).__init__(**kwargs)
        self.sparse_extraction = PartialMatrix(rate=rate, matrix_shape=matrix_shape, regularize=regularize)
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
      A condense operator, support tensor shape [batch, N, D, D],
    the operator will traverse dimension N, and perform a calculation on [batch, :, :, :].
    Example:
    >>>input_tensor = tf.ones((1, 16, 4, 4), dtype=tf.float32)
    >>>layer = Condense(num_capsule=8, matrix_shape=(4, 4))
    >>>out = layer(input_tensor)
    >>>print(out.shape)
    (1, 8, 4, 4)
    """
    def __init__(self, num_capsule: int, matrix_shape: tuple = (4, 4), regularize=1e-4, **kwargs):
        super(Condense, self).__init__(**kwargs)
        self.sparse_extraction = GlobalMatrix(num_capsule, matrix_shape, regularize=regularize)
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
    """
    Example:
    >>>input_tensor = tf.ones((1, 16, 4, 4), dtype=tf.float32)
    >>>layer = CapsFPNTiny(rate=(2, 2, 2, 1), matrix_shape=(4, 4))
    >>>out = layer(input_tensor)
    >>>print(out.shape)
    (1, 16, 4, 4)
    """

    def __init__(self, rate: Union[list, tuple] = None,
                 matrix_shape: Union[list, tuple] = None,
                 regularize=1e-5, **kwargs):
        super(CapsFPNTiny, self).__init__(**kwargs)
        rate = [2, 2, 2, 1] if rate is None else rate
        matrix_shape = (4, 4) if matrix_shape is None else matrix_shape
        self.condense1 = CondenseTiny(rate=rate[0], matrix_shape=matrix_shape, regularize=regularize)
        self.condense2 = CondenseTiny(rate=rate[1], matrix_shape=matrix_shape, regularize=regularize)
        self.condense3 = CondenseTiny(rate=rate[2], matrix_shape=matrix_shape, regularize=regularize)
        self.condense4 = CondenseTiny(rate=rate[3], matrix_shape=matrix_shape, regularize=regularize)
        self.feature_pyramid = layers.Concatenate(axis=-3)

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
    """
    Example:
    >>>input_tensor = tf.ones((1, 16, 4, 4), dtype=tf.float32)
    >>>layer = CapsFPNTiny(rate=(2, 2, 2, 1), matrix_shape=(4, 4))
    >>>out = layer(input_tensor)
    >>>print(out.shape)
    (1, 16, 4, 4)
    """
    def __init__(self, num_caps: Union[list, tuple] = None,
                 matrix_shape: Union[list, tuple] = None,
                 regularize=1e-4, **kwargs):
        super(CapsFPN, self).__init__(**kwargs)
        num_caps = (16, 8, 4, 4) if num_caps is None else num_caps
        matrix_shape = (4, 4) if matrix_shape is None else matrix_shape
        self.condense1 = Condense(num_caps[0], matrix_shape, regularize)
        self.condense2 = Condense(num_caps[1], matrix_shape, regularize)
        self.condense3 = Condense(num_caps[2], matrix_shape, regularize)
        self.condense4 = Condense(num_caps[3], matrix_shape, regularize)
        self.feature_pyramid = layers.Concatenate(axis=-3)

    def call(self, inputs, **kwargs):
        l1 = self.condense1(inputs)
        l2 = self.condense2(l1)
        l3 = self.condense3(l2)
        l4 = self.condense4(l3)
        pyramid = self.feature_pyramid([l1, l2, l3, l4])
        return pyramid
