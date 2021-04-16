import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from models import ETCModel
from utils import Dataset, plotHistory

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
# tf.config.experimental.set_visible_devices(gpus[6], 'GPU')
# tf.config.experimental.set_memory_growth(gpus[6], True)
# tf.config.experimental.set_visible_devices(gpus[7], 'GPU')
# tf.config.experimental.set_memory_growth(gpus[7], True)
# tf.config.experimental.set_visible_devices(gpus[8], 'GPU')
# tf.config.experimental.set_memory_growth(gpus[8], True)

# data_name = 'MNIST'
# data_name = 'MNIST_SHIFT'
# data_name = 'FASHION_MNIST'
# data_name = 'FASHION_MNIST_SHIFT'
data_name = 'CIFAR10'
# data_name = 'CIFAR10_SHIFT'
# data_name = 'SMALLNORB'

dataset = Dataset(data_name, config_path='config.json')
# batch_size = 128


# model = ETCModel(data_name=data_name, model_name='GHOSTNET')
model = ETCModel(data_name=data_name, model_name='CapsNet_Without_Decoder')
# model = ETCModel(data_name=data_name, model_name='RESNET_DWT50')

# resume weight you trained before
# model.load_graph_weights()  # load graph weights (bin folder)

history = model.train(dataset)

model.model.evaluate(dataset.X_test, dataset.y_test)

# 4.0 Plot history
plotHistory(history)

model.load_graph_weights()  # load graph weights (bin folder)
