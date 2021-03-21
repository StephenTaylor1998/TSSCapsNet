import tensorflow as tf
from utils import Dataset, plotHistory
from models import *


# some parameters
# model_name = 'DCT_Efficient_CapsNet'
# model_name = 'DCT_CapsNet_Attention'
# model_name = 'DCT_CapsNet_GumbelGate'
# model_name = 'DCT_CapsNet'
# model_name = 'DCT_Efficient_CapsNet'
model_name = 'RFFT_Efficient_CapsNet'
# model_name = 'Efficient_CapsNet'
# model_name = 'CapsNet'

# data_name = 'MNIST'
data_name = 'MNIST_SHIFT'

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[3], 'GPU')
tf.config.experimental.set_memory_growth(gpus[3], True)

# 1.0 Import the Dataset
dataset = Dataset(data_name, config_path='config.json')

# 2.0 Load the Model
model = TSSEfficientCapsNet(data_name, model_name=model_name, mode='train', verbose=True)

# 3.0 Train the Model
dataset_train, dataset_val = dataset.get_tf_data()
# history = model.train(dataset, initial_epoch=0)

# 4.0 Plot history
# plotHistory(history)

# 5.0 Load weights
model_test = TSSEfficientCapsNet(data_name, model_name='RFFT_Efficient_CapsNet', mode='test', verbose=True)
model.load_graph_weights()  # load graph weights (bin folder)

# 4.0 Test the Model

model_test.evaluate(dataset.X_test, dataset.y_test)  # if "smallnorb" use X_test_patch