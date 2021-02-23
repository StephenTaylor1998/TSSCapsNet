# Efficient-CapsNet Model Train
import tensorflow as tf

from models.model_zoo import DCTEfficientCapsNet
from utils import Dataset, plotImages, plotWrongImages, plotHistory
from models import EfficientCapsNet

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[3], 'GPU')
tf.config.experimental.set_memory_growth(gpus[3], True)

# some parameters

# model_name = 'MNIST'  # only MNIST is available
# data_name = 'MULTIMNIST'
data_name = 'MNIST_SHIFT'

# 1.0 Import the Dataset

dataset = Dataset(data_name, config_path='config.json')

# 1.1 Visualize imported dataset

n_images = 20  # number of images to be plotted
plotImages(dataset.X_test[:n_images, ..., 0], dataset.y_test[:n_images], n_images, dataset.class_names)

# 2.0 Load the Model

model_train = DCTEfficientCapsNet(data_name, mode='train', verbose=True)

# 3.0 Train the Model

dataset_train, dataset_val = dataset.get_tf_data()

history = model_train.train(dataset, initial_epoch=0)

plotHistory(history)
