import os
import tensorflow as tf


def load_model(target_dir):
    """

    :param target_dir: directory to load the model from
    :return: TensorFlow model
    """
    # sets the current working directory as the target_dir
    os.chdir(target_dir)

    # loads the model
    model = tf.keras.models.load_model('trained_model.h5')
    print(f'Loading model from {target_dir}\\trained_model.h5')

    return model
