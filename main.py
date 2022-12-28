import os

import ImageProcessing
import CNN

cwd = os.getcwd()

data_dir = cwd + '\\data\\brain_tumor_dataset'
cropped_dir = cwd + '\\data\\cropped_brain_tumor_dataset'
aug_data = cwd + '\\data\\aug_data'

# ImageProcessing.view_random_images(data_dir)

# ImageProcessing.plot_data_statistics(data_dir)

# ImageProcessing.crop_all_images(data_dir)

# ImageProcessing.view_random_images(cropped_dir)

# ImageProcessing.view_random_images(cropped_dir)

# CNN.split_data(cropped_dir)

# ImageProcessing.plot_train_test(aug_data)

training_data, validation_data, test_data = CNN.load_data(aug_data)

CNN.plot_train_val_test(training_data, validation_data, test_data)

CNN.build_model()



