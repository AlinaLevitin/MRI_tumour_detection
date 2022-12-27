"""
Splits the data to train, validation and test subsets randomly
"""
import os
import shutil

from sklearn.model_selection import train_test_split


def split_data(data_dir, test):

    print('Splitting data...')

    classes = ['no', 'yes']

    for label in classes:
        os.chdir(data_dir + '\\' + label)
        images = os.listdir(data_dir + '\\' + label)

        train_images, test_images = train_test_split(images, test_size=test, random_state=0)

        for train_image in train_images:
            original = data_dir + '\\' + label + '\\' + train_image
            target = os.path.dirname(os.getcwd()) + '\\aug_data\\train' + label + '\\' + train_image
            shutil.copyfile(original, target)
            print(f'copied {train_image} to {target}')

        for test_image in test_images:
            original = data_dir + '\\' + label + '\\' + test_image
            target = os.path.dirname(os.getcwd()) + '\\aug_data\\test' + label + '\\' + test_image
            shutil.copyfile(original, target)
            print(f'copied {test_image} to {target}')

