# coding=utf-8
import numpy as np
from tensorflow import keras
import tensorflow as tf
from tensorflow.python.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler

RANGE = 'range'
EXPONENT = 'exponent'


# class LearningRate(object):
#     def __init__(self, optimizer=None, method=None, lr_range=None, initial_lr=None):
#         self.optimizer = optimizer
#         self.method = method
#         self.lr_range = lr_range
#         self.initial_lr = initial_lr
#
#     def __call__(self, epoch, logs=None):
#         if self.optimizer is None:
#             raise ValueError('optimizer is none.')
#         if not hasattr(self.optimizer, 'learning_rate'):
#             raise ValueError('Optimizer must have a "learning_rate" attribute.')
#
#         # Get the current learning rate from model's optimizer.
#         lr = float(keras.backend.get_value(self.optimizer.learning_rate))
#         # Call schedule function to get the scheduled learning rate.
#         if self.method == 'range':
#             scheduled_lr = self.adjust_range(epoch, lr)
#         elif self.method == 'exponent':
#             scheduled_lr = self.adjust_exponent(epoch)
#         else:
#             scheduled_lr = lr
#
#         # Set the value back to the optimizer before this epoch starts
#         keras.backend.set_value(self.optimizer.learning_rate, scheduled_lr)
#
#     def adjust_range(self, epoch, lr):
#         if self.lr_range is None:
#             raise ValueError('lr_ranges is none.')
#         if epoch < self.lr_range[0][0] or epoch > self.lr_range[-1][0]:
#             return lr
#         for i in range(len(self.lr_range)-1, -1, -1):
#             if epoch >= self.lr_range[i][0]:
#                 return self.lr_range[i][1]
#         return lr
#
#     def adjust_exponent(self, epoch):
#         if self.initial_lr is None:
#             raise ValueError('initial_lr is none.')
#         if epoch < 10:
#             return self.initial_lr
#         else:
#             return self.initial_lr * tf.math.exp(0.01 * (10 - epoch))





# def learn_scheduler(lr_dec, lr):
#     def learning_scheduler_fn(epoch):
#         lr_new = lr * (lr_dec ** epoch)
#         return lr_new if lr_new >= 5e-5 else 5e-5
#
#     return learning_scheduler_fn
#
#
# lr_decay = tf.keras.callbacks.LearningRateScheduler(learn_scheduler(0.9, 0.001))
#
# reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
#     monitor='val_accuracy', factor=0.9,
#     patience=4, min_lr=0.000005, min_delta=0.0001, mode='max')


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
    # for Adam Optimizer
    lr = 1e-3
    if epoch > 360:
        lr = 1e-7
    elif epoch > 300:
        lr = 1e-6
    elif epoch > 240:
        lr = 1e-5
    elif epoch > 160:
        lr = 1e-4
    elif epoch > 80:
        lr = 1e-3
    print('Learning rate: ', lr)
    return lr

    # for SGD Optimizer
    # lr = 1e-1
    # if epoch > 360:
    #     lr = 1e-5
    # elif epoch > 300:
    #     lr = 1e-4
    # elif epoch > 240:
    #     lr = 1e-3
    # elif epoch > 160:
    #     lr = 1e-2
    # elif epoch > 80:
    #     lr = 1e-1
    # print('Learning rate: ', lr)
    # return lr


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
