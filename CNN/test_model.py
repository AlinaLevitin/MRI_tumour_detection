from sklearn.metrics import confusion_matrix
import numpy as np
import itertools
import tensorflow as tf
import matplotlib.pyplot as plt


def test_model(model, test_data):
    """
    function to test the model with test_data

    :param model: selected tensorflow.keras.sequential model
    :param test_data: test_data
    :return: test the accuracy on the test data and creates a confusion matrix
    """

    # evaluating the performance of the model with the test data
    model.evaluate(test_data)

    # making a confusion matrix
    figsize = (10, 10)

    # making prediction on the data
    y_pred = model.predict(test_data)

    # the true labels
    y_true = test_data.labels

    # confusion matrix
    cm = confusion_matrix(y_true, tf.round(y_pred))

    # normalizing the confusion matrix for percentages
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # setting the number of classes (should be 2)
    n_classes = cm[0].shape[0]

    # creating a new figure
    fig, ax = plt.subplots(figsize=figsize)

    # setting the color map (blue)
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)

    # setting up the labels (1 and 0)
    labels = np.arange(cm.shape[0])

    ax.set(title="Confusion Matrix",
           xlabel='Predicted Label',
           ylabel='True Label',
           xticks=np.arange(n_classes),
           yticks=np.arange(n_classes),
           xticklabels=labels,
           yticklabels=labels)

    # Set x-axis labels to bottom
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()

    # Adjust label size
    ax.yaxis.label.set_size(20)
    ax.xaxis.label.set_size(20)
    ax.title.set_size(20)

    # Set threshold for different colors
    threshold = (cm.max() + cm.min()) / 2.

    # Plot the text on each cell
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j] * 100:.1f}%)",
                 horizontalalignment="center",
                 color="white" if cm[i, j] > threshold else "black",
                 size=15)

    plt.show()
