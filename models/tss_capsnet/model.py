import os
import tensorflow as tf
from tensorflow.python.keras.utils.multi_gpu_utils import multi_gpu_model

from . import dct_capsnet_h1_graph_mnist
from . import dct_capsnet_e1_graph_mnist
from . import dct_capsnet_h1_attention_mnist
from . import dct_capsnet_h1_gumbel_gate_mnist
from . import dwt_capsnet_e1_graph_mnist
from . import rfft_capsnet_e1_graph_mnist
from . import wst_capsnet_e1_graph_mnist
from ..layers.model_base import Model

from utils.dataset import Dataset
from utils.tools import get_callbacks, marginLoss


class TSSCapsNet(Model):
    """
    A class used to manage the TSSCapsNet architecture.

    ...

    Attributes
    ----------
    data_name: str
        name of the model (only MNIST provided)
    mode: str
        model modality (Ex. 'test')
    config_path: str
        path configuration file
    verbose: bool
    n_routing: int
        number of routing interations

    Methods
    -------
    load_graph():
        load the network graph given the data_name
    train():
        train the constructed network with a given dataset. All train hyperparameters are defined in the configuration file
    """

    def __init__(self, data_name, model_name='DCT_CapsNet', mode='test', config_path='config.json', custom_path=None,
                 verbose=True, n_routing=3, gpu_number=None):
        Model.__init__(self, data_name, mode, config_path, verbose)
        self.model_name = model_name
        self.n_routing = n_routing
        self.load_config()
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
        self.load_graph()
        if gpu_number:
            self.model = multi_gpu_model(self.model, gpu_number)

    def load_graph(self):
        if self.data_name in ['MNIST', 'MNIST_SHIFT', 'FASHION_MNIST', 'FASHION_MNIST_SHIFT']:
            input_shape = self.config['MNIST_INPUT_SHAPE']
        elif self.data_name in ['CIFAR10', 'CIFAR10_SHIFT']:
            input_shape = self.config['CIFAR10_INPUT_SHAPE']
        else:
            raise NotImplemented

        if self.model_name == 'DCT_CapsNet':
            self.model = dct_capsnet_h1_graph_mnist.build_graph(input_shape, self.mode,
                                                                self.n_routing, self.verbose)
        elif self.model_name == 'DCT_CapsNet_GumbelGate':
            self.model = dct_capsnet_h1_gumbel_gate_mnist.build_graph(input_shape, self.mode,
                                                                      self.n_routing, self.verbose)
        elif self.model_name == 'DCT_CapsNet_Attention':
            self.model = dct_capsnet_h1_attention_mnist.build_graph(input_shape, self.mode,
                                                                    self.n_routing, self.verbose)
        else:
            raise NotImplemented

    def train(self, dataset=None, initial_epoch=0):
        callbacks = get_callbacks(self.model_name,
                                  self.tb_path,
                                  self.model_path_new_train,
                                  self.config['lr_dec'],
                                  self.config['lr'])

        if dataset == None:
            dataset = Dataset(self.data_name, self.config_path)
        dataset_train, dataset_val = dataset.get_tf_data()

        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.config['lr']),
                           loss=[marginLoss, 'mse'],
                           loss_weights=[1., self.config['lmd_gen']],
                           metrics={self.model_name: 'accuracy'})

        print('-' * 30 + f'{self.data_name} train' + '-' * 30)

        history = self.model.fit(dataset_train,
                                 epochs=self.config['epochs'],
                                 validation_data=(dataset_val),
                                 batch_size=self.config['batch_size'],
                                 initial_epoch=initial_epoch,
                                 callbacks=callbacks,
                                 workers=self.config['num_workers'])

        self.model.save_weights(os.path.join(self.config['saved_model_dir'],
                                             f"{self.model_name}",
                                             f"{self.model_name}_{self.data_name}.h5"))
        return history


class TSSEfficientCapsNet(Model):
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
                 custom_path=None, verbose=True, gpu_number=None):
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
        self.load_graph()
        if gpu_number:
            self.model = multi_gpu_model(self.model, gpu_number)

    def load_graph(self):
        if self.data_name in ['MNIST', 'MNIST_SHIFT', 'FASHION_MNIST', 'FASHION_MNIST_SHIFT']:
            input_shape = self.config['MNIST_INPUT_SHAPE']
        elif self.data_name in ['CIFAR10', 'CIFAR10_SHIFT']:
            input_shape = self.config['CIFAR10_INPUT_SHAPE']
        elif self.data_name == 'SMALLNORB':
            raise NotImplemented
            # self.model = efficient_capsnet_graph_smallnorb.build_graph(self.config['SMALLNORB_INPUT_SHAPE'],
            #                                                            self.mode,
            #                                                            self.verbose)
        elif self.data_name == 'MULTIMNIST':
            raise NotImplemented
            # self.model = efficient_capsnet_graph_multimnist.build_graph(self.config['MULTIMNIST_INPUT_SHAPE'],
            #                                                             self.mode, self.verbose)
        else:
            raise NotImplementedError

        if self.model_name == "DCT_Efficient_CapsNet":
            self.model = dct_capsnet_e1_graph_mnist.build_graph(input_shape, self.mode,
                                                                self.verbose)
        if self.model_name == "RFFT_Efficient_CapsNet":
            self.model = rfft_capsnet_e1_graph_mnist.build_graph(input_shape, self.mode,
                                                                 self.verbose)
        if self.model_name == "DWT_Efficient_CapsNet":
            self.model = dwt_capsnet_e1_graph_mnist.build_graph(input_shape, self.mode,
                                                                self.verbose)
        if self.model_name == "WST_Efficient_CapsNet":
            self.model = wst_capsnet_e1_graph_mnist.build_graph(input_shape, self.mode,
                                                                self.verbose)

    def train(self, dataset=None, initial_epoch=0):
        callbacks = get_callbacks(self.model_name,
                                  self.tb_path,
                                  self.model_path_new_train,
                                  self.config['lr_dec'],
                                  self.config['lr'])

        if dataset == None:
            dataset = Dataset(self.data_name, self.config_path)
        dataset_train, dataset_val = dataset.get_tf_data()

        if self.data_name == 'MULTIMNIST':
            self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.config['lr']),
                               loss=[marginLoss, 'mse', 'mse'],
                               loss_weights=[1., self.config['lmd_gen'] / 2, self.config['lmd_gen'] / 2],
                               metrics={'DCT_Efficient_CapsNet': 'accuracy'})
            steps = 10 * int(dataset.y_train.shape[0] / self.config['batch_size'])

        else:
            self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.config['lr']),
                               loss=[marginLoss, 'mse'],
                               loss_weights=[1., self.config['lmd_gen']],
                               metrics={self.model_name: 'accuracy'})
            steps = None

        print('-' * 30 + f'{self.data_name} train' + '-' * 30)

        history = self.model.fit(dataset_train,
                                 epochs=self.config[f'epochs'], steps_per_epoch=steps,
                                 validation_data=(dataset_val), batch_size=self.config['batch_size'],
                                 initial_epoch=initial_epoch,
                                 callbacks=callbacks,
                                 workers=self.config['num_workers'])

        self.model.save_weights(os.path.join(self.config['saved_model_dir'],
                                             f"{self.model_name}",
                                             f"{self.model_name}_{self.data_name}.h5"))

        return history
