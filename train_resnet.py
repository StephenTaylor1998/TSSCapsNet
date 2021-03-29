import os

import numpy as np

from models import ETCModel

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from utils import Dataset, plotHistory
from utils.pre_process_cifar10 import CIFAR_TRAIN_IMAGE_COUNT, image_rotate_random, image_shift_rand, \
    PARALLEL_INPUT_CALLS, image_squish_random, image_erase_random, generator

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[9], 'GPU')
tf.config.experimental.set_memory_growth(gpus[9], True)

# data_name = 'MNIST'
# data_name = 'MNIST_SHIFT'
# data_name = 'FASHION_MNIST'
# data_name = 'FASHION_MNIST_SHIFT'
data_name = 'CIFAR10'
# data_name = 'CIFAR10_SHIFT'

dataset = Dataset(data_name, config_path='config.json')
batch_size = 128


model = ETCModel(data_name=data_name, model_name='RESNET20')

# resume weight you trained before
model.load_graph_weights()  # load graph weights (bin folder)

history = model.train(dataset)

model.evaluate(dataset.X_test, dataset.y_test)

# 4.0 Plot history
plotHistory(history)

model.load_graph_weights()  # load graph weights (bin folder)
