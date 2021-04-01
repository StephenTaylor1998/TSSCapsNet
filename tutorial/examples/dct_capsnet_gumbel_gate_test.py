# DCT CapsNet Model Test
import tensorflow as tf
from utils import Dataset, plotImages, plotWrongImages
from models import TSSCapsNet

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

# some parameters
# data_name = 'MNIST'
# data_name = 'MNIST_SHIFT'
data_name = 'FASHION_MNIST_SHIFT'

n_routing = 3

# 1.0 Import the Dataset
dataset = Dataset(data_name, config_path='../../config.json')


# 1.1 Visualize imported dataset
n_images = 20  # number of images to be plotted
plotImages(dataset.X_test[:n_images, ..., 0], dataset.y_test[:n_images], n_images, dataset.class_names)

# 2.0 Load the Model
model_test = TSSCapsNet(data_name, model_name='DCT_CapsNet_GumbelGate', mode='test', verbose=True, n_routing=n_routing)
model_test.load_graph_weights()

# 3.0 Test the Model
model_test.evaluate(dataset.X_test, dataset.y_test)  # if "smallnorb" use X_test_patch
y_pred = model_test.predict(dataset.X_test)[0]  # if "smallnorb" use X_test_patch

# 3.1 Plot misclassified images
n_images = 20
plotWrongImages(dataset.X_test, dataset.y_test, y_pred,  # if "smallnorb" use X_test_patch
                n_images, dataset.class_names)
