import tensorflow as tf


def learn_scheduler(lr_dec, lr, warm_up=False):
    if warm_up:
        def learning_scheduler_fn(epoch):
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

    else:
        def learning_scheduler_fn(epoch):
            lr_new = lr * (lr_dec ** epoch)
            return lr_new if lr_new >= 5e-5 else 5e-5

    return learning_scheduler_fn


def get_callbacks(model_name, tb_log_save_path, saved_model_path, lr_dec, lr):
    tb = tf.keras.callbacks.TensorBoard(log_dir=tb_log_save_path, histogram_freq=0)
    lr_decay = tf.keras.callbacks.LearningRateScheduler(learn_scheduler(lr_dec, lr))

    if model_name == 'CapsNet':
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            saved_model_path, monitor='val_Original_CapsNet_accuracy:',
            save_best_only=False, save_weights_only=True, verbose=1)

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_Original_CapsNet_accuracy', factor=0.9,
            patience=4, min_lr=0.00001, min_delta=0.0001, mode='max')
        return [tb, model_checkpoint, reduce_lr]

    else:
        raise NotImplemented