import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from utils import Dataset, plotHistory
from models import TSSCapsNet
from models import TSSEfficientCapsNet
from models import EfficientCapsNet
from models import CapsNet


# model_name = 'DCT_CapsNet_Attention'      # TSSCapsNet
# model_name = 'DCT_CapsNet_GumbelGate'     # TSSCapsNet
# model_name = 'DCT_CapsNet'                # TSSCapsNet
# model_name = 'DCT_Efficient_CapsNet'      # TSSEfficientCapsNet
# model_name = 'RFFT_Efficient_CapsNet'     # TSSEfficientCapsNet
# model_name = 'DWT_Efficient_CapsNet'
model_name = "DWT_Multi_Attention_CapsNet"
# model_name = 'Efficient_CapsNet'          # EfficientCapsNet
# model_name = 'CapsNet'                    # CapsNet


# data_name = 'MNIST'
# data_name = 'MNIST_SHIFT'
# data_name = 'FASHION_MNIST'
# data_name = 'FASHION_MNIST_SHIFT'
data_name = 'CIFAR10'
# data_name = 'CIFAR10_SHIFT'

# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_visible_devices(gpus[9], 'GPU')
# tf.config.experimental.set_memory_growth(gpus[9], True)

# 1.0 Import the Dataset
dataset = Dataset(data_name, config_path='config.json')

# 2.0 Load the Model
# model_test = TSSCapsNet(data_name, model_name=model_name, mode='test', verbose=True)
model_test = TSSEfficientCapsNet(data_name, model_name=model_name, mode='test', verbose=True)
# model_test = EfficientCapsNet(data_name, model_name=model_name, mode='test', verbose=True)
# model_test = CapsNet(data_name, model_name=model_name, mode='test', verbose=True)

# 3.0 Load weights
model_test.load_graph_weights()  # load graph weights (bin folder)

# 4.0 Test the Model

# 4.1 Origin test set
model_test.evaluate(dataset.X_test, dataset.y_test)  # if "smallnorb" use X_test_patch

# 4.2 Shift test set(if using pre-process file end with 'shift.py')
train_dataset, test_dataset = dataset.get_tf_data()
model_test.model.compile(metrics=['accuracy'])
result = model_test.model.evaluate(test_dataset)
