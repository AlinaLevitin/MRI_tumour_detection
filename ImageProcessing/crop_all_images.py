import os

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import ImageProcessing


def crop_all_images(target_dir, target_classes=('yes', 'no')):
    """
    crops all images and saves them to new folder named "cropped_brain_tumor_dataset"
    :param target_dir: target directory (path)
    :param target_classes: selected classes to crop (default is both classes)
    :return cropping all images
    """

    print('Cropping all Images now...')

    # for both classes
    for target_class in target_classes:

        # making a list of all the images
        images = os.listdir(target_dir + '\\' + target_class)

        # cropping all images in the list
        for image in images:

            # reading the image
            image_read = mpimg.imread(target_dir + '/' + target_class + '/' + image)

            # showing the cropped image (was good to catch errors with some of the images)
            plt.axis('off')
            plt.imshow(ImageProcessing.crop_image(image_read))

            # setting the working directory
            os.chdir(target_dir)
            os.chdir("..")
            cwd = os.getcwd()

            # making a new directory for cropped images
            new_path = '/cropped_brain_tumor_dataset/' + target_class
            os.makedirs(cwd + new_path, exist_ok=True)
            os.chdir(cwd + new_path)

            # parsing the name of the image
            name = image.split('.')[0]

            # saving the image in the selected folder
            plt.savefig(f'{name}.png', format='png')

            print(f'cropped image {image}')

    print('Finished cropping all images')
