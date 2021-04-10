import tensorflow as tf

from utils.pre_process_cifar10 import CIFAR_TRAIN_IMAGE_COUNT, image_rotate_random, image_shift_rand, \
    PARALLEL_INPUT_CALLS, image_squish_random, generator, image_erase_random


def generate_tf_data(X_train, y_train, X_test, y_test, batch_size, for_capsule=True):
    dataset_train = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    dataset_train = dataset_train.shuffle(buffer_size=CIFAR_TRAIN_IMAGE_COUNT)

    if for_capsule:
        dataset_train = dataset_train.map(generator,
                                          num_parallel_calls=PARALLEL_INPUT_CALLS)
    dataset_train = dataset_train.batch(batch_size)
    dataset_train = dataset_train.prefetch(-1)

    dataset_test = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    dataset_test = dataset_test.cache()
    dataset_test = dataset_test.map(image_rotate_random)
    dataset_test = dataset_test.map(image_shift_rand,
                                    num_parallel_calls=PARALLEL_INPUT_CALLS)
    dataset_test = dataset_test.map(image_squish_random,
                                    num_parallel_calls=PARALLEL_INPUT_CALLS)
    dataset_test = dataset_test.map(image_erase_random,
                                    num_parallel_calls=PARALLEL_INPUT_CALLS)
    if for_capsule:
        dataset_test = dataset_test.map(generator,
                                        num_parallel_calls=PARALLEL_INPUT_CALLS)
    dataset_test = dataset_test.batch(batch_size)
    dataset_test = dataset_test.prefetch(-1)

    return dataset_train, dataset_test
