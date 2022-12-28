import matplotlib.pyplot as plt


def train_model(model, epochs, training_data, validation_data):
    history = model.fit(training_data,
                        epochs=epochs,
                        steps_per_epoch=len(training_data),
                        validation_data=validation_data,
                        validation_steps=len(validation_data)
                        )

    # plot the accuracy and loss vs epoch

    # making a new figure
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
