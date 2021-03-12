# Original CapsNet Model Train


import tensorflow as tf
from utils import Dataset, plotImages, plotWrongImages
from models import FFTCapsNet

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

# some parameters
# data_name = 'MNIST'
data_name = 'MNIST_SHIFT'

# n_routing = 3
n_routing = None

dataset = Dataset(data_name, config_path='config.json')  # only MNIST

# 1.1 Visualize imported dataset

n_images = 20  # number of images to be plotted
plotImages(dataset.X_test[:n_images, ..., 0], dataset.y_test[:n_images], n_images, dataset.class_names)

# 2.0 Load the Model

model_train = FFTCapsNet(data_name, model_name='DCT_CapsNet_Attention',
                         mode='train', verbose=True, n_routing=n_routing)

# 3.0 Train the Model

history = model_train.train(dataset, initial_epoch=0)
