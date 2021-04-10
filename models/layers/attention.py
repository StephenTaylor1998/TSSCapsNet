import math
import tensorflow as tf
# from tensorflow import keras
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


class MultiAttention(tf.keras.layers.Layer):
    """
    MultiAttention-block for capsule-format data.
    example:
        input_tensor ==>> [batch, caps_number, caps_length]
    Operator:
        final_output_tensor
    """

    def __init__(self, h, d, max_seq=2048, **kwargs):
        super(MultiAttention, self).__init__()
        self.len_k = None
        self.max_seq = None
        self.E = None
        self.h = h
        self.d = d
        self.dh = d // h
        self.Wq = layers.Dense(int(self.d // 2))
        self.Wk = layers.Dense(int(self.d // 2))
        self.Wv = layers.Dense(int(self.d))
        # self.Wq = keras.layers.Dense(int(self.d))
        # self.Wk = keras.layers.Dense(int(self.d))
        # self.Wv = keras.layers.Dense(int(self.d))
        self.fc = layers.Dense(d)
        self.max_seq = max_seq

    def build(self, input_shape):
        print(input_shape)
        self.len_k = input_shape[1]
        # self.max_seq = max(input_shape[0][1], input_shape[1][1], input_shape[2][1])

    def call(self, inputs, mask=None, weight_out=False, **kwargs):
        """
        :param inputs: a list of tensors. i.e) [Q, CapsSimilarity, V]
        :param mask: mask tensor
        :param weight_out: decide to get weather weight or not
        :param kwargs:
        :return: final tensor ( output of attention )
        """
        q = inputs
        print('input', q.shape)
        q = self.Wq(q)
        print('output', q.shape)
        q = tf.reshape(q, (q.shape[0], q.shape[1], self.h, -1))
        q = tf.transpose(q, (0, 2, 1, 3))  # batch, h, seq, dh
        print('transpose', q.shape)

        k = inputs
        k = self.Wk(k)
        k = tf.reshape(k, (k.shape[0], k.shape[1], self.h, -1))
        k = tf.transpose(k, (0, 2, 1, 3))

        v = inputs
        v = self.Wv(v)
        v = tf.reshape(v, (v.shape[0], v.shape[1], self.h, -1))
        v = tf.transpose(v, (0, 2, 1, 3))

        Kt = tf.transpose(k, [0, 1, 3, 2])
        # print('Q', q.shape, 'CapsSimilarity', Kt.shape)
        QKt = tf.matmul(q, Kt)
        # print("QKT", QKt.shape)
        logits = QKt
        logits = logits / math.sqrt(self.dh)

        if mask is not None:
            logits += (tf.cast(mask, tf.float32) * -1e9)

        # print('logits', logits.shape)
        attention_weights = tf.nn.softmax(logits, -1)
        # print('attention weights', attention_weights.shape)
        attention = tf.matmul(attention_weights, v)
        out = tf.transpose(attention, (0, 2, 1, 3))
        out = tf.reshape(out, (out.shape[0], -1, self.d))
        out = self.fc(out)
        return out


class BaselineAttention(layers.Layer):
    def __init__(self, h, d, max_seq=2048, **kwargs):
        super().__init__(**kwargs)
        self.h = h
        self.d = d
        self.dh = d // h
        self.Wq = layers.Dense(int(self.d // 2))
        self.Wk = layers.Dense(int(self.d // 2))
        self.Wv = layers.Dense(int(self.d))
        self.normal1 = layers.LayerNormalization()
        self.fc = layers.Dense(d)
        self.normal2 = layers.LayerNormalization()
        self.max_seq = max_seq
    
    def get_config(self):
        return super(BaselineAttention, self).get_config()

    def call(self, inputs, mask=None, weight_out=False, **kwargs):
        """
        :param inputs: a list of tensors. i.e) [Q, CapsSimilarity, V]
        :param mask: mask tensor
        :param weight_out: decide to get weather weight or not
        :param kwargs:
        :return: final tensor ( output of attention )
        """
        q = inputs
        q = self.Wq(q)
        q = tf.cast(q, tf.float32)
        q = tf.keras.layers.Reshape((q.shape[1], self.h, -1))(q)
        q = tf.transpose(q, (0, 2, 1, 3))  # batch, h, seq, dh

        k = inputs
        k = self.Wk(k)
        k = tf.keras.layers.Reshape((k.shape[1], self.h, -1))(k)
        k = tf.transpose(k, (0, 2, 1, 3))

        v = inputs
        v = self.Wv(v)
        v = tf.keras.layers.Reshape((v.shape[1], self.h, -1))(v)
        v = tf.transpose(v, (0, 2, 1, 3))

        Kt = tf.transpose(k, [0, 1, 3, 2])
        QKt = tf.matmul(q, Kt)
        logits = QKt
        logits = logits / math.sqrt(self.dh)

        if mask is not None:
            logits += (tf.cast(mask, tf.float32) * -1e9)

        attention_weights = tf.nn.softmax(logits, -1)
        attention = tf.matmul(attention_weights, v)

        out = tf.transpose(attention, (0, 2, 1, 3))
        out = tf.keras.layers.Reshape((-1, self.d))(out)
        out = self.normal1(out)
        out = self.fc(out)
        out = self.normal2(out)

        return out
