import math
from tensorflow.keras import layers


class FFC(layers.Layer):
    """
      "Folding Full Connection": A Linear like operator, support tensor shape [batch, N, D],
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
    def __init__(self, out_length, kernel_size=1, strides=1, group=1, as_matrix=True):
        """
        "Folding Full Connection"
        input: [batch, N, D] ==>> output: [batch, N, out_length]
        :param out_length: the length of last dimension.
        :param kernel_size: conv1d kernel_size.
        :param strides: conv1d strides.
        :param group: conv1d group.
        :param as_matrix: if as matrix, group will be sqrt(D) automatically.
        """
        super(FFC, self).__init__()
        self.out_numbers = out_length
        self.kernel_size = kernel_size
        self.strides = strides
        self.group = group
        self.as_matrix = as_matrix
        self.folding_full_connection = None

    def build(self, input_shape):
        if self.as_matrix:
            self.group = int(math.sqrt(input_shape[-1]))
            assert self.group * self.group == input_shape[-1], "[ERROR] input_shape[-1] should = group * group."

        assert self.group <= self.out_numbers, "[ERROR] self.group should <= self.out_numbers"

        self.folding_full_connection = layers.Conv1D(self.out_numbers, self.kernel_size, strides=self.strides,
                                                     groups=self.group, use_bias=False)

    def call(self, inputs, **kwargs):
        out = self.folding_full_connection(inputs)
        return out


class Condense(layers.Layer):
    """
      A conv1d operator, support tensor shape [batch, N, D],
    the operator will traverse dimension N, and perform a calculation on [batch, n:n+rate, :].
    It is recommended to use rate == strings to avoid redundant computation.
    Example:
        parameter(input_shape=[1, 3, 4], out_length=2, rate=2, strides=2)
        X:  1  1  1  1      W:                              Y:
            1  1  1  1          1  1  1  1 | 2  2  2  2         12 | 12
            2  2  2  2          2  2  2  2 | 1  1  1  1         24 | 24
            2  2  2  2
    Output:
        [batch, N/rate, out_length]
    """

    def __init__(self, out_length, rate=2, strides=2):
        super(Condense, self).__init__()
        self.sparse_extraction = layers.Conv1D(out_length, rate, strides=strides, use_bias=False)

    def call(self, inputs, **kwargs):
        out = self.sparse_extraction(inputs)
        return out
