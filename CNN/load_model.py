import os
import tensorflow as tf


def load_model(target_dir):

    print(f'Loading model from {target_dir}/trained_model')
    os.chdir(target_dir)
    model = tf.keras.models.load_model('trained_model')

    return model
