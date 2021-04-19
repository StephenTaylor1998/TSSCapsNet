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

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.utils.multi_gpu_utils import multi_gpu_model

from utils.dataset import Dataset
from utils.get_resnet_layer import get_resnet_depth_from_name
from . import dwt_resnet_capsule_with_fpn_routing
from . import mobilenet_v2_cifar
from . import resnet_cifar
from . import resnet_cifar_dwt
from .call_backs import get_callbacks
from ..layers.model_base import Model


class ETCModel(Model):

    def __init__(self, data_name, model_name='DCT_Efficient_CapsNet', mode='test', config_path='config.json',
                 custom_path=None, verbose=True, gpu_number=None, optimizer='Adam', half_filter_in_resnet=True,
                 use_tiny_block=True, heterogeneous=False, **kwargs):
        Model.__init__(self, data_name, mode, config_path, verbose)
        self.model_name = model_name
        if custom_path is not None:
            self.model_path = custom_path
        else:
            self.model_path = os.path.join(self.config['saved_model_dir'],
                                           f"{self.model_name}",
                                           f"{self.model_name}_{self.data_name}.h5")

        os.makedirs(os.path.join(self.config['saved_model_dir'], f"{self.model_name}"), exist_ok=True)

        self.model_path_new_train = os.path.join(self.config['saved_model_dir'],
                                                 f"{self.model_name}",
                                                 f"{self.model_name}_{self.data_name}_{'{epoch:03d}'}.h5")
        self.tb_path = os.path.join(self.config['tb_log_save_dir'], f"{self.model_name}_{self.data_name}")
        self.half = half_filter_in_resnet
        self.tiny = use_tiny_block
        self.heterogeneous = heterogeneous
        self.load_graph()
        if gpu_number:
            self.model = multi_gpu_model(self.model, gpu_number)
        self.optimizer = optimizer

    def load_graph(self):
        if self.data_name in ['MNIST', 'MNIST_SHIFT', 'FASHION_MNIST', 'FASHION_MNIST_SHIFT']:
            input_shape = self.config['MNIST_INPUT_SHAPE']
            num_classes = 10
        elif self.data_name in ['CIFAR10', 'CIFAR10_SHIFT']:
            input_shape = self.config['CIFAR10_INPUT_SHAPE']
            num_classes = 10
        elif self.data_name == 'SMALLNORB':
            input_shape = self.config['SMALLNORB_INPUT_SHAPE']
            num_classes = 5
        elif self.data_name == 'MULTIMNIST':
            raise NotImplemented
        else:
            raise NotImplemented

        if self.model_name.startswith("RESNET"):
            # example "RESNET18_DWT_Tiny_Half" "RESNET50_DWT_Tiny_Half"
            half = True if "Half" in self.model_name else False
            tiny = True if "Tiny" in self.model_name else False
            if "DWT" in self.model_name:
                self.model = resnet_cifar_dwt.build_graph(
                    input_shape, num_classes, depth=get_resnet_depth_from_name(self.model_name), half=half, tiny=tiny)
            else:
                self.model = resnet_cifar.build_graph(
                    input_shape, num_classes, depth=get_resnet_depth_from_name(self.model_name), half=half)

        elif self.model_name == "MOBILENETv2":
            self.model = mobilenet_v2_cifar.build_graph(input_shape, num_classes)
        elif self.model_name.startswith("DWT_") and self.model_name.endswith("_FPN_CIFAR"):
            # example: "DWT_Tiny_Half_R18_Tiny_FPN_CIFAR"
            half = True if "Half_R" in self.model_name else False
            tiny = True if "DWT_Tiny" in self.model_name else False
            if "Tiny_FPN_CIFAR" in self.model_name:
                routing_name_list = ["Tiny_FPN", "Tiny_FPN", "Tiny_FPN"]
            elif "Attention_FPN_CIFAR" in self.model_name:
                routing_name_list = ['Attention', 'Attention', 'Attention']
            elif "FPN_CIFAR" in self.model_name:
                routing_name_list = ['FPN', 'FPN', 'FPN']
            else:
                print("FPN type is not support!")
                raise NotImplementedError

            self.model = dwt_resnet_capsule_with_fpn_routing.build_graph(
                input_shape, num_classes=10, routing_name_list=routing_name_list, regularize=1e-4, tiny=tiny, half=half,
                depth=get_resnet_depth_from_name(self.model_name), heterogeneous=self.heterogeneous)

        else:
            raise NotImplemented

    def train(self, dataset=None, initial_epoch=0, resume=False):
        callbacks = get_callbacks(self.model_path_new_train, optimizer=self.optimizer)

        if dataset is None:
            dataset = Dataset(self.data_name, self.config_path)
        dataset_train, dataset_val = dataset.get_tf_data(for_capsule=False)

        if self.optimizer == 'SGD':
            self.model.compile(optimizer=tf.keras.optimizers.SGD(lr=self.config['ETC_MODEL_LR'], momentum=0.9),
                               loss='categorical_crossentropy',
                               metrics='accuracy')
        elif self.optimizer == 'Adam':
            self.model.compile(optimizer=tf.keras.optimizers.Adam(lr=self.config['ETC_MODEL_LR']),
                               loss='categorical_crossentropy',
                               metrics='accuracy')
        else:
            print("optimizer must be select in ['Adam', 'SGD']")
            raise ValueError

        steps = None
        if resume:
            self.load_graph_weights()

        print('-' * 30 + f'{self.data_name} train' + '-' * 30)

        history = self.model.fit(dataset_train,
                                 epochs=self.config[f'ETC_MODEL_EPOCHS'], steps_per_epoch=steps,
                                 validation_data=dataset_val, batch_size=self.config['batch_size'],
                                 initial_epoch=initial_epoch,
                                 callbacks=callbacks,
                                 workers=self.config['num_workers'])

        self.model.save_weights(os.path.join(self.config['saved_model_dir'],
                                             f"{self.model_name}",
                                             f"{self.model_name}_{self.data_name}.h5"))

        return history

    def evaluate(self, X_test, y_test, dataset_name="Test"):
        print('-' * 30 + f'{self.data_name} Evaluation' + '-' * 30)

        y_pred = self.model.predict(X_test)
        acc = np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1)) / y_test.shape[0]
        test_error = 1 - acc
        print(f"{dataset_name} acc:", acc)
        print(f"{dataset_name} error [%]: {test_error :.4%}")
        if self.data_name == "MULTIMNIST":
            print(
                f"N° misclassified images: {int(test_error * len(y_test) * self.config['n_overlay_multimnist'])} "
                f"out of {len(y_test) * self.config['n_overlay_multimnist']}")
        else:
            print(f"N° misclassified images: {int(test_error * len(y_test))} out of {len(y_test)}")
