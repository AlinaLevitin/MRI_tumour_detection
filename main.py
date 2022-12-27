import os

import ImageProcessing

cwd = os.getcwd()

data_dir = cwd + '\\data\\brain_tumor_dataset'
cropped_dir = cwd + '\\data\\cropped_brain_tumor_dataset'

# ImageProcessing.view_random_images(data_dir)

# ImageProcessing.plot_data_statistics(data_dir)

ImageProcessing.crop_all_images(data_dir)

# ImageProcessing.view_random_images(cropped_dir)

