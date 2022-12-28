import numpy as np
import matplotlib.pyplot as plt


def plot_train_val_test(training_data, validation_data, test_data):
    # get the value counts
    train_no = np.unique(training_data.classes, return_counts=True)[1][0]
    train_yes = np.unique(training_data.classes, return_counts=True)[1][1]

    val_no = np.unique(validation_data.classes, return_counts=True)[1][0]
    val_yes = np.unique(validation_data.classes, return_counts=True)[1][1]

    test_no = np.unique(test_data.classes, return_counts=True)[1][0]
    test_yes = np.unique(test_data.classes, return_counts=True)[1][1]

    # set up the data and the labels
    values_no = [train_no, val_no, test_no]
    values_yes = [train_yes, val_yes, test_yes]

    labels = ['train', 'validation', 'test']

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