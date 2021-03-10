import os
import tensorflow as tf

from ..layers.model_base import Model
from . import original_capsnet_graph_mnist

from utils.dataset import Dataset
from utils.tools import get_callbacks, marginLoss


class CapsNet(Model):
    """
    A class used to manage the original CapsNet architecture.

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

    def __init__(self, data_name, model_name='CapsNet', mode='test', config_path='config.json', custom_path=None,
                 verbose=True, n_routing=3):
        Model.__init__(self, data_name, mode, config_path, verbose)
        self.model_name = model_name
        self.n_routing = n_routing
        self.load_config()
        if custom_path != None:
            self.model_path = custom_path
        else:
            # "original_capsnet_{}.{}.h5".format(self.data_name, "{epoch:02d}")
            self.model_path = os.path.join(self.config['saved_model_dir'],
                                           f"{self.model_name}_{self.data_name}.h5")
        self.model_path_new_train = os.path.join(self.config['saved_model_dir'],
                                                 f"{self.model_name}_{self.data_name}_{'{epoch:02d}'}.h5")
        self.tb_path = os.path.join(self.config['tb_log_save_dir'], f"{self.model_name}_{self.data_name}")
        self.load_graph()

    def load_graph(self):
        self.model = original_capsnet_graph_mnist.build_graph(self.config['MNIST_INPUT_SHAPE'], self.mode,
                                                              self.n_routing, self.verbose)

    def train(self, dataset=None, initial_epoch=0):
        callbacks = get_callbacks(self.model_name,
                                  self.tb_path,
                                  self.model_path_new_train,
                                  self.config['lr_dec'],
                                  self.config['lr'])

        if dataset == None:
            dataset = Dataset(self.data_name, self.config_path)
        dataset_train, dataset_val = dataset.get_tf_data()

        # self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.config['lr']),
        #                    loss=[marginLoss, 'mse'],
        #                    loss_weights=[1., self.config['lmd_gen']],
        #                    metrics={'Original_CapsNet': multiAccuracy})

        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.config['lr']),
                           loss=[marginLoss, 'mse'],
                           loss_weights=[1., self.config['lmd_gen']],
                           metrics={'Original_CapsNet': 'accuracy'})

        print('-' * 30 + f'{self.data_name} train' + '-' * 30)

        history = self.model.fit(dataset_train,
                                 epochs=self.config['epochs'],
                                 validation_data=(dataset_val),
                                 batch_size=self.config['batch_size'],
                                 initial_epoch=initial_epoch,
                                 callbacks=callbacks,
                                 workers=self.config['num_workers'])

        self.model.save_weights(os.path.join(self.config['saved_model_dir'],
                                             f"{self.model_name}_{self.data_name}.h5"))

        return history