import os

import ImageProcessing

cwd = os.getcwd()

target_dir = cwd + '\\data\\brain_tumor_dataset'

# ImageProcessing.view_random_images(target_dir)

# ImageProcessing.plot_data_statistics(target_dir)

# ImageProcessing.crop_all_images(target_dir)

ImageProcessing.view_random_images(cwd + '\\data\\cropped_brain_tumor_dataset')

