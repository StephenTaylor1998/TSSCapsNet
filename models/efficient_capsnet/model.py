import os
import tensorflow as tf
from ..layers.model_base import Model
from . import efficient_capsnet_graph_multimnist
from . import efficient_capsnet_graph_mnist
from . import efficient_capsnet_graph_smallnorb
from utils.dataset import Dataset
from utils.tools import get_callbacks, marginLoss


class EfficientCapsNet(Model):
    """
    A class used to manage an Efficiet-CapsNet model. 'data_name' and 'mode' define the particular architecure and modality of the
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

    def __init__(self, data_name, mode='test', model_name='Efficient_CapsNet', config_path='config.json',
                 custom_path=None, verbose=True):
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

    def load_graph(self):
        if self.data_name == 'MNIST' or self.data_name == 'MNIST_SHIFT':
            self.model = efficient_capsnet_graph_mnist.build_graph(self.config['MNIST_INPUT_SHAPE'], self.mode,
                                                                   self.verbose)
        elif self.data_name == 'SMALLNORB':
            self.model = efficient_capsnet_graph_smallnorb.build_graph(self.config['SMALLNORB_INPUT_SHAPE'], self.mode,
                                                                       self.verbose)
        elif self.data_name == 'MULTIMNIST':
            self.model = efficient_capsnet_graph_multimnist.build_graph(self.config['MULTIMNIST_INPUT_SHAPE'],
                                                                        self.mode, self.verbose)

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
                               metrics={'Efficient_CapsNet': 'accuracy'})
            steps = 10 * int(dataset.y_train.shape[0] / self.config['batch_size'])
        else:
            self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.config['lr']),
                               loss=[marginLoss, 'mse'],
                               loss_weights=[1., self.config['lmd_gen']],
                               metrics={'Efficient_CapsNet': 'accuracy'})
            steps = None

        print('-' * 30 + f'{self.data_name} train' + '-' * 30)

        history = self.model.fit(dataset_train,
                                 epochs=self.config[f'epochs'],
                                 steps_per_epoch=steps,
                                 validation_data=(dataset_val),
                                 batch_size=self.config['batch_size'],
                                 initial_epoch=initial_epoch,
                                 callbacks=callbacks,
                                 workers=self.config['num_workers'])

        self.model.save_weights(os.path.join(self.config['saved_model_dir'],
                                             f"{self.model_name}",
                                             f"{self.model_name}_{self.data_name}.h5"))

        return history
