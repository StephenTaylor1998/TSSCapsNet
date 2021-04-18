from tensorflow.python.keras.callbacks import ModelCheckpoint, LearningRateScheduler


def lr_schedule_adam(epoch):
    """Learning Rate Schedule
    # Arguments
        epoch (int): The number of epochs
    # Returns
        lr (float32): learning rate
    """
    # for Adam Optimizer
    lr = 1e-3

    if epoch > 350:
        lr = 1e-5
    elif epoch > 300:
        lr = 1e-4
    elif epoch > 200:
        lr = 2e-4
    elif epoch > 100:
        lr = 5e-4
    print('Learning rate: ', lr)
    return lr


def lr_schedule_sgd(epoch):
    """Learning Rate Schedule
    # Arguments
        epoch (int): The number of epochs
    # Returns
        lr (float32): learning rate
    """
    lr = 1e-1
    if epoch > 350:
        lr = 5e-4
    elif epoch > 300:
        lr = 1e-3
    elif epoch > 250:
        lr = 5e-3
    elif epoch > 200:
        lr = 1e-2
    elif epoch > 100:
        lr = 5e-2
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


def get_callbacks(weight_path, optimizer='Adam'):
    checkpoint = ModelCheckpoint(filepath=weight_path,
                                 monitor='val_accuracy',
                                 verbose=1,
                                 save_best_only=True)
    if optimizer == 'Adam':
        lr_scheduler = LearningRateScheduler(lr_schedule_adam)
    elif optimizer == 'SGD':
        lr_scheduler = LearningRateScheduler(lr_schedule_sgd)
    else:
        raise NotImplemented

    callbacks = [checkpoint, lr_scheduler]
    return callbacks
