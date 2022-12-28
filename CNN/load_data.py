import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def load_data(target_dir):
    """
    load data with augmentation for training and validation subsets (no augments for test data)
    :param target_dir: target directory (path)
    :return: training_data, validation_data and test_data
    """
    tf.random.set_seed(42)

    # preprocess data (scale)
    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       validation_split=0.2,
                                       shear_range=0.1,
                                       brightness_range=[0.5, 1.5],
                                       horizontal_flip=True,
                                       vertical_flip=True
                                       )

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    # Import data from directories and turn it into batches with augmented data
    training_data = train_datagen.flow_from_directory(directory=target_dir + '\\train',
                                                      subset="training",
                                                      batch_size=32,
                                                      target_size=(224, 224),
                                                      class_mode='binary',
                                                      seed=42)
    validation_data = train_datagen.flow_from_directory(directory=target_dir + '\\train',
                                                        subset="validation",
                                                        batch_size=32,
                                                        target_size=(224, 224),
                                                        class_mode='binary',
                                                        seed=42)

    test_data = test_datagen.flow_from_directory(directory=target_dir + '\\test',
                                                 batch_size=32,
                                                 target_size=(224, 224),
                                                 class_mode='binary',
                                                 seed=42)

    return training_data, validation_data, test_data
