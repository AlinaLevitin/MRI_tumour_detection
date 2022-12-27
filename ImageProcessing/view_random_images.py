"""
function to view a bulk of images of both classes
"""
import os

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random


def view_random_images(target_dir, target_classes=('yes', 'no'), img_num=25):
    """
    Shows two figures with 25 random images from selected target directory for both classes

    :param target_dir: target directory (path)
    :param target_classes: selected classes to show (default is both classes)
    :param img_num: number of images to show
    """

    # making a figure for each class
    for target_class in target_classes:

        # set the target directory
        target_folder = target_dir + '\\' + target_class

        # Set up a figure
        plt.figure()

        # Get random image paths according to the number of images chosen in img_num
        random_images = random.sample(os.listdir(target_folder), img_num)

        # Read in the image and plot it using matplotlib
        for index, image in enumerate(random_images):
            plt.subplot(5, 5, index + 1)
            plt.imshow(mpimg.imread(target_folder + "\\" + image))

            # removing the axis
            plt.axis('off')

        # setting the title according to the class
        plt.suptitle(f'Tumor: {target_class}', fontsize=20)

        # showing the figure
        plt.show()
