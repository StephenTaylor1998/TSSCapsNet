# Copyright 2021 Hang-Chi Shen. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

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
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        saved_model_path, monitor=f"val_{model_name}_accuracy",
        save_best_only=True, save_weights_only=True, verbose=1)
    return [tb, model_checkpoint, lr_decay]


