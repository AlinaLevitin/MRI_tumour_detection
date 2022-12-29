import os
import shutil

from sklearn.model_selection import train_test_split


def split_data(data_dir, test=0.2):
    """
    Splits the data to train, validation and test subsets randomly
    :param data_dir: directory location of the data
    :param test: the portion of the data to use as test (default is 0.2)
    :return: split the data to test and train
    """

    print('Deleting old files...')
    os.chdir(data_dir)
    target_dir = os.path.dirname(os.getcwd()) + '\\aug_data'
    os.makedirs(target_dir, exist_ok=True)

    for files in os.listdir(target_dir):
        path = os.path.join(target_dir, files)
        try:
            shutil.rmtree(path)
        except OSError:
            os.remove(path)
    print('Finished deleting files')

    print('Splitting data...')

    classes = ['no', 'yes']

    for label in classes:
        os.chdir(data_dir + '\\' + label)
        images = os.listdir(data_dir + '\\' + label)

        train_images, test_images = train_test_split(images, test_size=test, random_state=0)

        for train_image in train_images:
            os.makedirs(os.path.dirname(os.path.dirname(os.getcwd())) + '\\aug_data\\train\\' + label, exist_ok=True)
            original = data_dir + '\\' + label + '\\' + train_image
            target = os.path.dirname(os.path.dirname(os.getcwd())) + '\\aug_data\\train\\' + label + '\\' + train_image
            shutil.copy2(src=original, dst=target)
            print(f'copied {train_image} to {target}')

        for test_image in test_images:
            os.makedirs(os.path.dirname(os.path.dirname(os.getcwd())) + '\\aug_data\\test\\' + label, exist_ok=True)
            original = data_dir + '\\' + label + '\\' + test_image
            target = os.path.dirname(os.path.dirname(os.getcwd())) + '\\aug_data\\test\\' + label + '\\' + test_image
            shutil.copy2(original, target)
            print(f'copied {test_image} to {target}')

    print('Finished splitting data')

