"""
BUILD CNN
"""
import tensorflow as tf


def build_model(input_shape=(224, 224, 3)):
    """
    Set up the model using keras.Sequential

    :param input_shape: image input shape (default is 224, 224, 3)
    :return: tf.keras.models.Sequential model
    """

    # construct the model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=10, kernel_size=3, activation='relu', input_shape=input_shape),
        tf.keras.layers.Conv2D(filters=10, kernel_size=3, activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=2, padding='valid'),
        tf.keras.layers.Conv2D(filters=10, kernel_size=3, activation='relu'),
        tf.keras.layers.Conv2D(filters=10, kernel_size=3, activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=2, padding='valid'),
        tf.keras.layers.Conv2D(filters=10, kernel_size=3, activation='relu'),
        tf.keras.layers.Conv2D(filters=10, kernel_size=3, activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=2, padding='valid'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])

    return model
