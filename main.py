import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from models import CapsNet
from models import TSSCapsNet
from models import EfficientCapsNet
from models import TSSEfficientCapsNet
from utils import Dataset, plotHistory
from utils.argparse import get_terminal_args

if __name__ == '__main__':

    args = get_terminal_args()

    if args.select_gpu:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_visible_devices(gpus[args.select_gpu], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[args.select_gpu], True)
    gpu_number = None

    # 1.0 Import the Dataset
    dataset = Dataset(args.data_name, config_path='config.json')

    # 2.0 Load the Model
    if args.arch == "CapsNet":
        model = CapsNet(args.data_name, model_name=args.model_name,
                        mode='test', verbose=True, gpu_number=gpu_number)
    elif args.arch == "EfficientCapsNet":
        model = EfficientCapsNet(args.data_name, model_name=args.model_name,
                                 mode='test', verbose=True, gpu_number=gpu_number)
    elif args.arch == "TSSCapsNet":
        model = TSSCapsNet(args.data_name, model_name=args.model_name,
                           mode='test', verbose=True, gpu_number=gpu_number)
    elif args.arch == "TSSEfficientCapsNet":
        model = TSSEfficientCapsNet(args.data_name, model_name=args.model_name,
                                    mode='test', verbose=True, gpu_number=gpu_number)
    else:
        raise NotImplementedError

    # 3.0 Train the Model
    dataset_train, dataset_val = dataset.get_tf_data()
    history = model.train(dataset, args.initial_epoch)

    # 4.0 Plot history
    plotHistory(history)

    # 5.0 Load weights
    model.load_graph_weights()  # load graph weights (bin folder)

    # 6.0 Test the Model
    model.evaluate(dataset.X_test, dataset.y_test)  # if "smallnorb" use X_test_patch
