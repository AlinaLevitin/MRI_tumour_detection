import os
import matplotlib.pyplot as plt


def plot_data_statistics(target_dir):
    """
    plot the number of images im a bar graph

    :param target_dir: target_dir: target directory (path)
    :return plots data statistics
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


def plot_train_test(target_dir):
    """
    plot the number of images in a bar graph after splitting to test and train
    :param target_dir: target directory
    :return: plots the data distribution
    """
    # get the value counts
    train_no = len(os.listdir(target_dir + '\\train\\no'))
    train_yes = len(os.listdir(target_dir + '\\train\\yes'))

    test_no = len(os.listdir(target_dir + '\\test\\no'))
    test_yes = len(os.listdir(target_dir + '\\test\\yes'))

    # set up the data and the labels
    values_no = [train_no, test_no]
    values_yes = [train_yes, test_yes]

    labels = ['train', 'test']

    # create a new figure
    fig, ax = plt.subplots(figsize=(7, 10))
    fig.suptitle('Data distribution', fontsize=20)

    # set up the column width
    width = 0.6

    # plot the data
    ax.bar(labels, values_no, label='no', width=width)
    ax.bar(labels, values_yes, label='yes', width=width / 2, align="edge")

    # make title, and axis
    ax.set_title(f'{sum(values_no) + sum(values_yes)} images', fontsize=15)
    ax.set_ylabel('Counts', fontsize=15)
    plt.tick_params(labelsize=15)
    plt.legend()

    plt.show()
