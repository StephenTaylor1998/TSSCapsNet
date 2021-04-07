# coding=utf-8
import numpy as np
from tensorflow import keras
import tensorflow as tf
from tensorflow.python.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler

# RANGE = 'range'
# EXPONENT = 'exponent'


def lr_schedule(epoch):
    """Learning Rate Schedule
    # Arguments
        epoch (int): The number of epochs
    # Returns
        lr (float32): learning rate
    >>>lr = 1e-3
    >>>if epoch > 180:
    >>>    lr *= 0.5e-3
    >>>elif epoch > 160:
    >>>    lr *= 1e-3
    >>>elif epoch > 120:
    >>>    lr *= 1e-2
    >>>elif epoch > 80:
    >>>    lr *= 1e-1
    >>>print('Learning rate: ', lr)
    >>>return lr
    """
    # # for SGD Optimizer (about 10 hours on single GPU)
    # lr = 1e-1
    # if epoch > 600:
    #     lr = 5e-6
    # elif epoch > 550:
    #     lr = 1e-5
    # elif epoch > 550:
    #     lr = 5e-5
    # elif epoch > 500:
    #     lr = 1e-4
    # elif epoch > 450:
    #     lr = 5e-4
    # elif epoch > 400:
    #     lr = 1e-3
    # elif epoch > 350:
    #     lr = 5e-3
    # elif epoch > 300:
    #     lr = 1e-2
    # elif epoch > 200:
    #     lr = 5e-2
    # print('Learning rate: ', lr)
    # return lr

    # for Adam Optimizer
    lr = 1e-3

    if epoch > 240:
        lr = 1e-6
    elif epoch > 120:
        lr = 1e-5
    elif epoch > 60:
        lr = 1e-4
    print('Learning rate: ', lr)
    return lr


def learning_scheduler_fn(epoch):
    lr = 1e-3
    lr_dec = 0.97
    # warm_up
    if epoch < 10:
        lr_new = (lr - 1e-6) / 10 * epoch + 1e-6
        return lr_new
    # lr maintenance
    elif epoch < 30:
        return lr
    # lr decay
    lr_new = lr * (lr_dec ** (epoch - 30))
    return lr_new if lr_new >= 1e-6 else 1e-6


def get_callbacks(weight_path):
    checkpoint = ModelCheckpoint(filepath=weight_path,
                                 monitor='val_accuracy',
                                 verbose=1,
                                 save_best_only=True)

    lr_scheduler = LearningRateScheduler(lr_schedule)

    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=5,
                                   min_lr=0.5e-6)
    callbacks = [checkpoint, lr_reducer, lr_scheduler]
    return callbacks
