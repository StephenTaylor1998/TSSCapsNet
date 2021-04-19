# Copyright 2021 Hang-Chi Shen. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from utils import Dataset, plotHistory
from models import TSSCapsNet
from models import EfficientCapsNet
from models import CapsNet
from models import ETCModel


# model_name = 'DCT_CapsNet_Attention'      # TSSCapsNet
# model_name = 'DCT_CapsNet_GumbelGate'     # TSSCapsNet
# model_name = 'DCT_CapsNet'                # TSSCapsNet
# model_name = 'DCT_Efficient_CapsNet'      # TSSCapsNet
# model_name = 'RFFT_Efficient_CapsNet'     # TSSCapsNet
# model_name = 'DWT_Efficient_CapsNet'
# model_name = "DWT_Multi_Attention_CapsNet"
# model_name = 'Efficient_CapsNet'           # EfficientCapsNet
# model_name = 'CapsNet'                     # CapsNet
# model_name = 'DWT_Caps_FPN'
model_name = 'RESNET_DWT50_Tiny'


# data_name = 'MNIST'
# data_name = 'MNIST_SHIFT'
# data_name = 'FASHION_MNIST'
# data_name = 'FASHION_MNIST_SHIFT'
data_name = 'CIFAR10'
# data_name = 'CIFAR10_SHIFT'
# data_name = 'SMALLNORB'

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[5], 'GPU')
tf.config.experimental.set_memory_growth(gpus[5], True)
gpu_number = None

# 1.0 Import the Dataset
dataset = Dataset(data_name, config_path='config.json')

# 2.0 Load the Model
# model = TSSCapsNet(data_name, model_name=model_name, mode='train', verbose=True, gpu_number=gpu_number)
# model = EfficientCapsNet(data_name, model_name=model_name, mode='train', verbose=True, gpu_number=gpu_number)
# model = CapsNet(data_name, model_name=model_name, mode='train', verbose=True, gpu_number=gpu_number)
model = ETCModel(data_name=data_name, model_name=model_name)

# 3.0 Train the Model
# dataset_train, dataset_val = dataset.get_tf_data(for_capsule=False)
# dataset_train, dataset_val = dataset.get_tf_data()
history = model.train(dataset, initial_epoch=0)

# 4.0 Plot history
plotHistory(history)
