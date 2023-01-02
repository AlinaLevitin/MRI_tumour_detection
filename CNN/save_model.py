import os


def save_model(model, target_dir):
    """

    :param model: TensorFlow model to save
    :param target_dir: directory to save the model at
    :return: saves the model
    """
    # sets the current working directory as the target_dir
    os.chdir(target_dir)

    # saves the model
    model.save('trained_model.h5')
    print(f'Model was saved to {target_dir}\\trained_model.h5')
