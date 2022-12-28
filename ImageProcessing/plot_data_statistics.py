import os
import matplotlib.pyplot as plt


def plot_data_statistics(target_dir):
    """
    plot the number of images im a bar graph

    :param target_dir: target_dir: target directory (path)
    """
    # collecting all images in both classes
    yes_images = os.listdir(target_dir + '\\yes')
    no_images = os.listdir(target_dir + '\\no')

    # the number of images in each class
    num_yes_images = len(yes_images)
    num_no_images = len(no_images)

    # making a figure for the plot
    plt.figure(figsize=(10, 7))

    # bar graph
    plt.bar(('Positive for Tumor', 'Negative for Tumor'), (num_yes_images, num_no_images))

    # setting font size and labels
    plt.title('Brain MRI images', fontsize=20)
    plt.ylabel('Count', fontsize=15)
    plt.tick_params(labelsize=15)

    # showing the plot
    plt.show()
