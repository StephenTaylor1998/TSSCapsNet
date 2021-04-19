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
import tensorflow as tf
from tensorflow.python.keras.utils.multi_gpu_utils import multi_gpu_model

from utils.get_resnet_layer import get_resnet_depth_from_name
from . import dct_capsnet_e1_graph_mnist

from . import dct_capsnet_h1_attention_mnist
from . import dct_capsnet_h1_graph_mnist
from . import dct_capsnet_h1_gumbel_gate_mnist
from . import dwt_capsnet_e1_graph_mnist
from . import dwt_capsnet_e1_graph_smallnorb
from . import dwt_capsnet_fpn_graph_mnist
from . import dwt_capsnet_fpn_graph_smallnorb
from . import dwt_resnet_capsnet_fpn_graph_cifar
from . import rfft_capsnet_e1_graph_mnist
from . import wst_capsnet_e1_graph_mnist

from .call_backs import get_callbacks
from ..etc_model.call_backs import get_callbacks as etc_callback
from ..layers.model_base import Model
from utils.dataset import Dataset
from utils.tools import marginLoss


class TSSCapsNet(Model):

    def __init__(self, data_name, model_name='DCT_Efficient_CapsNet', mode='test', config_path='config.json',
                 custom_path=None, verbose=True, gpu_number=None, optimizer='Adam', half_filter_in_resnet=True,
                 use_tiny_block=True, heterogeneous=False, **kwargs):
        Model.__init__(self, data_name, mode, config_path, verbose)
        self.model_name = model_name
        if custom_path != None:
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
        elif self.data_name == 'SMALLNORB_INPUT_SHAPE':
            num_classes = 5
            input_shape = self.config['CIFAR10_INPUT_SHAPE']
        elif self.data_name == 'MULTIMNIST':
            raise NotImplemented
        else:
            raise NotImplementedError

        if self.model_name == "DCT_E_MNIST":
            self.model = dct_capsnet_e1_graph_mnist.build_graph(input_shape, self.mode, self.model_name)
        elif self.model_name == "DCT_H_A_MNIST":
            self.model = dct_capsnet_h1_attention_mnist.build_graph(input_shape, self.mode, 3, self.model_name)
        elif self.model_name == "DCT_H_MNIST":
            self.model = dct_capsnet_h1_graph_mnist.build_graph(input_shape, self.mode, 3, self.model_name)
        elif self.model_name == "DCT_H_Gumbel_MNIST":
            self.model = dct_capsnet_h1_gumbel_gate_mnist.build_graph(input_shape, self.mode, 3, self.model_name)
        elif self.model_name == "DWT_E_MNIST":
            self.model = dwt_capsnet_e1_graph_mnist.build_graph(input_shape, self.mode, self.model_name)
        elif self.model_name == "DWT_E_SMALLNORB":
            self.model = dwt_capsnet_e1_graph_smallnorb.build_graph(input_shape, self.mode, self.model_name)
        elif self.model_name == "DWT_FPN_MNIST":
            self.model = dwt_capsnet_fpn_graph_mnist.build_graph(input_shape, self.mode, num_classes,
                                                                 ['FPN', 'FPN', 'FPN'], regularize=1e-4,
                                                                 name=self.model_name)
        elif self.model_name == "DWT_Tiny_FPN_MNIST":
            self.model = dwt_capsnet_fpn_graph_mnist.build_graph(input_shape, self.mode, num_classes,
                                                                 ['FPN', 'FPN', 'FPN'], regularize=1e-4,
                                                                 name=self.model_name)
        elif self.model_name == "DWT_Attention_FPN_MNIST":
            self.model = dwt_capsnet_fpn_graph_mnist.build_graph(input_shape, self.mode, num_classes,
                                                                 ['FPN', 'FPN', 'FPN'], regularize=1e-4,
                                                                 name=self.model_name)
        elif self.model_name == "DWT_FPN_SMALLNORB":
            self.model = dwt_capsnet_fpn_graph_smallnorb.build_graph(input_shape, self.mode, num_classes,
                                                                     ['FPN', 'FPN', 'FPN'], regularize=1e-4,
                                                                     name=self.model_name)
        elif self.model_name == "DWT_Tiny_FPN_SMALLNORB":
            self.model = dwt_capsnet_fpn_graph_smallnorb.build_graph(input_shape, self.mode, num_classes,
                                                                     ['FPN', 'FPN', 'FPN'], regularize=1e-4,
                                                                     name=self.model_name)
        elif self.model_name == "DWT_Attention_FPN_SMALLNORB":
            self.model = dwt_capsnet_fpn_graph_smallnorb.build_graph(input_shape, self.mode, num_classes,
                                                                     ['FPN', 'FPN', 'FPN'], regularize=1e-4,
                                                                     name=self.model_name)
        elif self.model_name == "RFFT_E_MNIST":
            self.model = rfft_capsnet_e1_graph_mnist.build_graph(input_shape, self.mode, self.model_name)
        elif self.model_name == "WST_E_MNIST":
            self.model = wst_capsnet_e1_graph_mnist.build_graph(input_shape, self.mode, self.model_name)
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

            self.model = dwt_resnet_capsnet_fpn_graph_cifar.build_graph(
                input_shape, self.mode, num_classes=10, routing_name_list=routing_name_list, regularize=1e-4,
                depth=get_resnet_depth_from_name(self.model_name), tiny=tiny, half=half, name=self.model_name,
                heterogeneous=self.heterogeneous
            )
        else:
            print(f"model name {self.model_name} is NotImplemented")
            raise NotImplemented

    def train(self, dataset=None, initial_epoch=0):
        callbacks = get_callbacks(self.model_name,
                                  self.tb_path,
                                  self.model_path_new_train,
                                  self.config['lr_dec'],
                                  self.config['lr'],
                                  optimizer=self.optimizer)

        if dataset is None:
            dataset = Dataset(self.data_name, self.config_path)
        dataset_train, dataset_val = dataset.get_tf_data()

        if self.optimizer == 'Adam':
            self.model.compile(optimizer=tf.keras.optimizers.Adam(lr=self.config['lr'], momentum=0.9),
                               loss=[marginLoss, 'mse'],
                               loss_weights=[1., self.config['lmd_gen']],
                               metrics={self.model_name: 'accuracy'})
        else:
            self.model.compile(optimizer=tf.keras.optimizers.SGD(lr=self.config['lr']),
                               loss=[marginLoss, 'mse'],
                               loss_weights=[1., self.config['lmd_gen']],
                               metrics={self.model_name: 'accuracy'})

        # self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.config['lr']),
        #                    loss=[marginLoss, 'mse'],
        #                    loss_weights=[1., self.config['lmd_gen']],
        #                    metrics={self.model_name: 'accuracy'})
        steps = None

        print('-' * 30 + f'{self.data_name} train' + '-' * 30)

        history = self.model.fit(dataset_train,
                                 epochs=self.config[f'epochs'], steps_per_epoch=steps,
                                 validation_data=dataset_val, batch_size=self.config['batch_size'],
                                 initial_epoch=initial_epoch,
                                 callbacks=callbacks,
                                 workers=self.config['num_workers'])

        self.model.save_weights(os.path.join(self.config['saved_model_dir'],
                                             f"{self.model_name}",
                                             f"{self.model_name}_{self.data_name}.h5"))

        return history
