import json
import numpy as np
from tqdm.notebook import tqdm
from utils.tools import multiAccuracy
from utils import pre_process_multimnist


class Model(object):
    """
    A class used to share common model functions and attributes.

    ...

    Attributes
    ----------
    data_name: str
        name of the data (Ex. 'MNIST')
    mode: str
        model modality (Ex. 'test')
    config_path: str
        path configuration file
    verbose: bool

    Methods
    -------
    load_config():
        load configuration file
    load_graph_weights():
        load network weights
    predict(dataset_test):
        use the model to predict dataset_test
    evaluate(X_test, y_test):
        comute accuracy and test error with the given dataset (X_test, y_test)
    save_graph_weights():
        save model weights
    """

    def __init__(self, data_name, mode='test', config_path='config.json', verbose=True):
        self.data_name = data_name
        self.model = None
        self.mode = mode
        self.config_path = config_path
        self.config = None
        self.verbose = verbose
        self.load_config()

    def load_config(self):
        """
        Load config file
        """
        with open(self.config_path) as json_data_file:
            self.config = json.load(json_data_file)

    def load_graph_weights(self):
        try:
            self.model.load_weights(self.model_path)
        except Exception as e:
            print("[ERROR] Graph Weights not found '{}'".format(self.model_path))

    def predict(self, dataset_test):
        return self.model.predict(dataset_test)

    def evaluate(self, X_test, y_test):
        print('-' * 30 + f'{self.data_name} Evaluation' + '-' * 30)
        if self.data_name == "MULTIMNIST":
            dataset_test = pre_process_multimnist.generate_tf_data_test(X_test, y_test, self.config["shift_multimnist"],
                                                                        n_multi=self.config['n_overlay_multimnist'])
            acc = []
            for X, y in tqdm(dataset_test, total=len(X_test)):
                y_pred, X_gen1, X_gen2 = self.model.predict(X)
                acc.append(multiAccuracy(y, y_pred))
            acc = np.mean(acc)
        else:
            y_pred, X_gen = self.model.predict(X_test)
            acc = np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1)) / y_test.shape[0]
        test_error = 1 - acc
        print('Test acc:', acc)
        print(f"Test error [%]: {(test_error):.4%}")
        if self.data_name == "MULTIMNIST":
            print(
                f"N° misclassified images: {int(test_error * len(y_test) * self.config['n_overlay_multimnist'])} out of {len(y_test) * self.config['n_overlay_multimnist']}")
        else:
            print(f"N° misclassified images: {int(test_error * len(y_test))} out of {len(y_test)}")

    def save_graph_weights(self):
        self.model.save_weights(self.model_path)