# %% md

# Efficient-CapsNet Model Test


import tensorflow as tf
from utils import Dataset, plotImages, plotWrongImages
from models import TSSEfficientCapsNet

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[3], 'GPU')
tf.config.experimental.set_memory_growth(gpus[3], True)

# some parameters
# data_name = 'MULTIMNIST'
data_name = 'MNIST_SHIFT'

custom_path = None  # if you've trained a new model, insert here the full graph weights path

# 1.0 Import the Dataset


dataset = Dataset(data_name, config_path='config.json')

# 1.1 Visualize imported dataset


n_images = 20  # number of images to be plotted
plotImages(dataset.X_test[:n_images, ..., 0], dataset.y_test[:n_images], n_images, dataset.class_names)

# 2.0 Load the Model

model_test = TSSEfficientCapsNet(data_name, model_name='RFFT_Efficient_CapsNet', mode='test', verbose=True, custom_path=custom_path)

model_test.load_graph_weights()  # load graph weights (bin folder)

# 3.0 Test the Model


model_test.evaluate(dataset.X_test, dataset.y_test)  # if "smallnorb" use X_test_patch

# 3.1 Plot misclassified images


# not working with MultiMNIST

y_pred = model_test.predict(dataset.X_test)[0]  # if "smallnorb" use X_test_patch

n_images = 20
plotWrongImages(dataset.X_test, dataset.y_test, y_pred,  # if "smallnorb" use X_test_patch
                n_images, dataset.class_names)
