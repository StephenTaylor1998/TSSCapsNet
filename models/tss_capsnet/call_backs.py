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

    if model_name == 'DCT_CapsNet':
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            saved_model_path, monitor='val_DCT_CapsNet_accuracy',
            # saved_model_path, monitor='val_DCT_CapsNet_loss',
            save_best_only=True, save_weights_only=True, verbose=1)
        return [tb, model_checkpoint, lr_decay]

    elif model_name == 'DCT_CapsNet_GumbelGate':
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            saved_model_path, monitor='val_DCT_CapsNet_GumbelGate_accuracy',
            # saved_model_path, monitor='val_DCT_CapsNet_GumbelGate_loss',
            save_best_only=True, save_weights_only=True, verbose=1)
        return [tb, model_checkpoint, lr_decay]

    elif model_name == 'DCT_Efficient_CapsNet':
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            saved_model_path, monitor='val_DCT_Efficient_CapsNet_accuracy',
            # saved_model_path, monitor='val_DCT_Efficient_CapsNet_loss',
            save_best_only=True, save_weights_only=True, verbose=1)
        return [tb, model_checkpoint, lr_decay]

    elif model_name == 'DCT_CapsNet_Attention':
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            saved_model_path, monitor='val_DCT_CapsNet_Attention_accuracy',
            # saved_model_path, monitor='val_DCT_CapsNet_Attention_loss',
            save_best_only=True, save_weights_only=True, verbose=1)
        return [tb, model_checkpoint, lr_decay]

    elif model_name == 'RFFT_Efficient_CapsNet':
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            saved_model_path, monitor='val_RFFT_Efficient_CapsNet_accuracy',
            # saved_model_path, monitor='val_RFFT_Efficient_CapsNet_loss',
            save_best_only=True, save_weights_only=True, verbose=1)
        return [tb, model_checkpoint, lr_decay]

    elif model_name == 'DWT_Efficient_CapsNet':
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            saved_model_path, monitor='val_DWT_Efficient_CapsNet_accuracy',
            # saved_model_path, monitor='val_DWT_Efficient_CapsNet_loss',
            save_best_only=True, save_weights_only=True, verbose=1)
        return [tb, model_checkpoint, lr_decay]

    elif model_name == 'DWT_Multi_Attention_CapsNet':
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            saved_model_path, monitor='val_DWT_Multi_Attention_CapsNet_accuracy',
            # saved_model_path, monitor='val_DWT_Multi_Attention_CapsNet_loss',
            save_best_only=True, save_weights_only=True, verbose=1)
        return [model_checkpoint, lr_decay]
        # return [tb, model_checkpoint, lr_decay]

    elif model_name == 'WST_Efficient_CapsNet':
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            saved_model_path, monitor='val_WST_Efficient_CapsNet_accuracy',
            # saved_model_path, monitor='val_WST_Efficient_CapsNet_loss',
            save_best_only=True, save_weights_only=True, verbose=1)
        return [tb, model_checkpoint, lr_decay]

    else:
        raise NotImplemented