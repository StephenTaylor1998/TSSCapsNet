import os

import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from utils import Dataset, plotHistory
from utils.pre_process_cifar10 import CIFAR_TRAIN_IMAGE_COUNT, image_rotate_random, image_shift_rand, \
    PARALLEL_INPUT_CALLS, image_squish_random, image_erase_random, generator

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[9], 'GPU')
tf.config.experimental.set_memory_growth(gpus[9], True)

# data_name = 'MNIST'
# data_name = 'MNIST_SHIFT'
# data_name = 'FASHION_MNIST'
# data_name = 'FASHION_MNIST_SHIFT'
data_name = 'CIFAR10'
# data_name = 'CIFAR10_SHIFT'

dataset = Dataset(data_name, config_path='config.json')
batch_size = 128

resnet50 = tf.keras.applications.ResNet50(
    input_shape=(32, 32, 3),
    classes=10,
    include_top=True,
    weights=None
)

resnet50.compile(optimizer=tf.optimizers.Adam(),
                 # loss=tf.losses.SparseCategoricalCrossentropy(),
                 loss=tf.losses.CategoricalCrossentropy(),
                 metrics=['accuracy'])
# resnet50 = tf.keras.applications.ResNet50()
dataset_train, dataset_test = dataset.get_tf_data(for_capsule=False)


def learn_scheduler(lr_dec, lr):
    def learning_scheduler_fn(epoch):
        lr_new = lr * (lr_dec ** epoch)
        return lr_new if lr_new >= 5e-5 else 5e-5

    return learning_scheduler_fn


lr_decay = tf.keras.callbacks.LearningRateScheduler(learn_scheduler(0.9, 0.001))

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_accuracy', factor=0.9,
    patience=4, min_lr=0.000005, min_delta=0.0001, mode='max')

history = resnet50.fit(dataset_train, epochs=100, workers=16, batch_size=batch_size,
                       validation_data=dataset_test,
                       callbacks=[lr_decay]
                       )

resnet50.evaluate(x_test, y_test)

# 4.0 Plot history
plotHistory(history)
