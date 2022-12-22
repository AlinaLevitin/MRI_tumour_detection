import os

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import tensorflow as tf

from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.image import ImageDataGenerator


cwd = os.getcwd()
path = cwd + "\\data\\brain_tumor_dataset"


classes = os.listdir(path)

# print(classes)


# noinspection PyStatementEffect
def view_random_image(target_dir, target_class):
    # set the target directory
    target_folder = target_dir + "/" + target_class

    # Get a random image path
    random_image = random.sample(os.listdir(target_folder), 1)
    print(random_image)

    # Read in the image and plot it using matplotlib
    image = mpimg.imread(target_folder + "/" + random_image[0])
    plt.imshow(image)
    plt.title(target_class)
    plt.axis('off')
    plt.show()

    # show the shape
    print(f"Image shape: {image.shape}")

    return image


# rand_image = view_random_image(target_dir=path, target_class="yes")

# PREPROCESSING AND LOADING DATA:

# set seed
tf.random.set_seed(42)

# preprocess data (scale)
datagen = ImageDataGenerator(rescale=1./255)


# Setup paths to our data directory

dir_path = cwd + "\\data\\brain_tumor_dataset\\"

# Import data from directories and trun it into batches with augmented data
data = datagen.flow_from_directory(directory=dir_path, batch_size=32, target_size=(224, 224), class_mode='binary', seed=42)

# train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# BUILD CNN:

# construct the model
model_1 = tf.keras.models.Sequential([
                                      tf.keras.layers.Conv2D(filters=10, kernel_size=3, activation='relu', input_shape=(224, 224, 3)),
                                      tf.keras.layers.Conv2D(filters=10, kernel_size=3, activation='relu'),
                                      tf.keras.layers.MaxPool2D(pool_size=2, padding='valid'),
                                      tf.keras.layers.Conv2D(filters=10, kernel_size=3, activation='relu'),
                                      tf.keras.layers.Conv2D(filters=10, kernel_size=3, activation='relu'),
                                      tf.keras.layers.MaxPool2D(pool_size=2, padding='valid'),
                                      tf.keras.layers.Flatten(),
                                      tf.keras.layers.Dense(1, activation='sigmoid')
                                      ])

# Compile the model
model_1.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])

# fit the data
history_1 = model_1.fit(data, epochs=5, steps_per_epoch=len(data))
