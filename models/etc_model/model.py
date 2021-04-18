import os

import tensorflow as tf
from tensorflow.python.keras.utils.multi_gpu_utils import multi_gpu_model

from utils.dataset import Dataset
from . import capsule_with_fpn_routing
from . import ghostnet_cifar
from . import mobilenet_v2_cifar
from . import resnet_cifar
from . import resnet_cifar_dwt
from .call_backs import get_callbacks
from ..layers.model_base import Model


class ETCModel(Model):
    """
    A class used to manage an DCT-Efficiet-CapsNet model. 'data_name' and 'mode' define the particular architecure and modality of the
    generated network.
    ...
    Attributes
    ----------
    data_name: str
        name of the model (Ex. 'MNIST')
    mode: str
        model modality (Ex. 'test')
    config_path: str
        path configuration file
    custom_path: str
        custom weights path
    verbose: bool
    Methods
    -------
    load_graph():
        load the network graph given the data_name
    train(dataset, initial_epoch)
        train the constructed network with a given dataset. All train hyperparameters are defined in the configuration file
    """

    def __init__(self, data_name, model_name='DCT_Efficient_CapsNet', mode='test', config_path='config.json',
                 custom_path=None, verbose=True, gpu_number=None, optimizer='Adam', half_filter_in_resnet=True):
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

        if self.model_name == "RESNET18":
            self.model = resnet_cifar.build_graph(input_shape, num_classes, depth=18, half=self.half)
        elif self.model_name == "RESNET34":
            self.model = resnet_cifar.build_graph(input_shape, num_classes, depth=34, half=self.half)
        elif self.model_name == "RESNET50":
            self.model = resnet_cifar.build_graph(input_shape, num_classes, depth=50, half=self.half)
        elif self.model_name == "RESNET_DWT18":
            self.model = resnet_cifar_dwt.build_graph(input_shape, num_classes, depth=18, half=self.half)
        elif self.model_name == "RESNET_DWT34":
            self.model = resnet_cifar_dwt.build_graph(input_shape, num_classes, depth=34, half=self.half)
        elif self.model_name == "RESNET_DWT50":
            self.model = resnet_cifar_dwt.build_graph(input_shape, num_classes, depth=50, half=self.half)
        elif self.model_name == "RESNET_DWT18_Tiny":
            self.model = resnet_cifar_dwt.build_graph(input_shape, num_classes, depth=18, half=self.half)
        elif self.model_name == "RESNET_DWT34_Tiny":
            self.model = resnet_cifar_dwt.build_graph(input_shape, num_classes, depth=34, half=self.half)
        elif self.model_name == "RESNET_DWT50_Tiny":
            self.model = resnet_cifar_dwt.build_graph(input_shape, num_classes, depth=50, half=self.half)
        elif self.model_name == "GHOSTNET":
            self.model = ghostnet_cifar.build_graph(input_shape, num_classes)
        elif self.model_name == "MOBILENETv2":
            self.model = mobilenet_v2_cifar.build_graph(input_shape, num_classes)
        elif self.model_name == "DWT_Caps_FPN":
            self.model = capsule_with_fpn_routing.build_graph(
                input_shape, num_classes, ['FPN', 'FPN', 'FPN'])
        elif self.model_name == "DWT_Caps_FPNTiny":
            self.model = capsule_with_fpn_routing.build_graph(
                input_shape, num_classes, ['FPNTiny', 'FPNTiny', 'FPNTiny'])
        elif self.model_name == "DWT_Caps_Attention":
            self.model = capsule_with_fpn_routing.build_graph(
                input_shape, num_classes, ['Attention', 'Attention', 'Attention'])
        elif self.model_name == "CapsNet_Without_Decoder":
            self.model = capsule_with_fpn_routing.build_graph(
                input_shape, num_classes, ['FPN', 'FPN', 'FPN'])
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
