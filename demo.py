from models.layers.gate import GateModule
import tensorflow as tf

batch = 2
height = 4
width = 4
channel = 8
tensor1 = tf.ones((batch, height, width, channel))

input_tensor = tf.keras.Input(shape=(height, width, channel))
sampled, sample_key = GateModule()(tensor1)
print(sampled)
print(sampled.shape)
print(sample_key)

