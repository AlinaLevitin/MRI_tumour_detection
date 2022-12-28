import os
import matplotlib.pyplot as plt


def train_model(model, epochs, training_data, validation_data, target_dir):
    """
    function to train the model with training data and validation data
    :param model: selected tensorflow.keras.sequential model
    :param epochs: number of epochs to train
    :param training_data: training data
    :param validation_data: validation data
    :param target_dir: directory to save the model at
    :return: trains the model and creates a plot of the accuracy and loss vs epoch
    """

    # training the model
    history = model.fit(training_data,
                        epochs=epochs,
                        steps_per_epoch=len(training_data),
                        validation_data=validation_data,
                        validation_steps=len(validation_data)
                        )
    os.chdir(target_dir)
    model.save('trained_model')
    print(f'Model was saved to {target_dir+ "/trained_model.h5"}')

    # plot the accuracy and loss vs epoch
    plt.figure(figsize=(8, 8))

    # setting the x ticks
    ticks = [i for i in range(1, epochs + 1)]

    # plotting the accuracy and loss from the training
    plt.plot(ticks, history.history['accuracy'], label='accuracy', marker='o')
    plt.plot(ticks, history.history['loss'], label='loss', marker='o')
    plt.plot(ticks, history.history['val_accuracy'], label='validation accuracy', marker='o')
    plt.plot(ticks, history.history['val_loss'], label='validation loss', marker='o')

    # setting labels and title
    plt.title('Accuracy and Loss vs Epoch', fontsize=20)
    plt.xlabel('epoch', fontsize=15)
    plt.ylabel('accuracy and loss [a.u]', fontsize=15)
    plt.xticks(ticks, ticks)

    # showing the legend
    plt.legend()

    # showing the plot
    plt.show()
