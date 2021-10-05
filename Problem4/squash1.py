# Inspired by https://deeplearning.lipingyang.org/wp-content/uploads/2016/12/Building-powerful-image-classification-models-using-very-little-data.pdf
import inspect
import itertools
import os
import json
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from tensorflow.keras.backend import clear_session
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Input, Dense, Flatten, Activation, Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.metrics import binary_accuracy
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.preprocessing.image import load_img, ImageDataGenerator
from tensorflow import reduce_mean
from sklearn.model_selection import ParameterGrid


#
def get_xy(data_dir, target, preprocess_func):
    """
    Load the raw image data into a data matrix, X, and target vector, y.

    :param data_dir: The root directory of the data
    :param target: The target (subdirectory of the data)
    :param preprocess_func: a function that takes on input 'x' and returns the preprocessed data
    :return: the data matrix 'x' and target vector 'y'
    """
    x = []
    y = []
    for label in range(2):
        subdir = os.path.join(data_dir, target, str(label))
        for file_name in os.listdir(subdir):
            image_file = os.path.join(subdir, file_name)
            im = np.array(load_img(image_file))
            x.append(im)
            y.append(label)
    x = np.array(x)
    y = np.array(y)
    x = preprocess_func(x)
    return x, y


#
def preprocess(x, target_size=(150, 150)):
    """
    This is a preprocess function meant to be tuned using input arguments. Once you
    determine the best preprocessing approach, make sure to save the keyword arguments
    in preprocess_params.json to submit to Web-CAT. The example "main" code does this.

    :param x: The data matrix (n, 240, 240, 3) numpy array uint8
    :param target_size: Resize the images to this shape
    :return: The updated data matrix a numpy array (n, num_rows, num_columns, num_channels)
    """
    # make sure any imports you need are imported here.
    import tensorflow as tf
    x = tf.image.resize(x, size=target_size).numpy()
    # make sure to use .numpy() on tensorflow objects to get the numpy array
    # x = tf.image.rgb_to_grayscale(x).numpy()
    # x = tf.image.adjust_contrast(x, 10).numpy()
    x = tf.image.per_image_standardization(x)
    return x


#
def build_fn(input_shape, optimizer, loss, output_activation, metrics):
    """
    Custom model build function can take any parameters you want to build a network
    for your model.

    :param input_shape: the shape of each sample (image)
    :param optimizer: the optimizer function
    :param loss: the loss function
    :param output_activation: the output activation function
    :param metrics: other metrics to track
    :return: a compiled model ready to 'fit'
    """
    # make sure to clear any previous nodes in the computation graph to save memory
    clear_session()

    # Fully connected linear model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1, activation=output_activation))
    model.summary()
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model


#
def analyze_results(output_path, data_dir, target, preprocess_func):
    """
    Helper function to compute train/validation performance and show some of the mistakes.

    :param output_path: path to folder containing trained model
    :param data_dir: where to find the train/validation data
    :param target: 'train' or 'valid'
    :param preprocess_func: the preprocessing function that takes one parameter 'x'
    :return: a dictionary with train/valid loss/accuracy
    """
    clear_session()
    model_file = os.path.join(output_path, 'model.h5')
    # get best model and update JSON with results
    model = load_model(model_file, compile=True)

    x, y = get_xy(data_dir, target, preprocess_func)
    y_hat = model.predict(x).reshape((-1,))

    loss = reduce_mean(binary_crossentropy(y_true=y, y_pred=y_hat)).numpy()
    acc = reduce_mean(binary_accuracy(y_true=y, y_pred=y_hat)).numpy()

    # update the params dictionary to include train/validation performance
    d2 = {
        f'{target}_loss': float(loss),
        f'{target}_acc': float(acc),
    }
    print(d2)

    # image grid of misclassified examples
    for s in ['fp', 'fn']:
        if s == 'fp':
            # false positives
            index = y_hat > y
        elif s == 'fn':
            # false negatives
            index = y_hat < y
        else:
            raise ValueError(f'Unknown mistake: {s}')
        mistakes = x[index]
        num_mistakes = len(mistakes)
        index = np.random.choice(range(len(mistakes)), size=min(16, len(mistakes)), replace=len(mistakes) < 16)
        mistakes = mistakes[index]
        fig = plt.figure(figsize=(8, 8))
        grid = ImageGrid(fig, 111, nrows_ncols=(4, 4), axes_pad=0.1)
        for i, ax in enumerate(grid):
            im = mistakes[i]
            im = im - im.min(axis=(0, 1))
            im = im / im.max(axis=(0, 1))
            if im.shape[2] == 2:
                im = np.concatenate((im, np.zeros(im.shape[:2] + (1,))), axis=2)
            if i < len(mistakes):
                ax.imshow(im)
        png_file = os.path.join(output_path, f'{target}_{num_mistakes}_{s}_{loss:.6f}_{100 * acc:.1f}.png')
        plt.savefig(png_file)
        plt.close(fig)
    return d2


def assemble_results(output_root, param_grid_names):
    """
    Helper function to traverse output root directory to assemble and save a CSV file with results.

    :param output_root: The directory that contains all the output model directories
    :param param_grid_names: The names of the parameter sets stored in JSON files
    :return: None
    """
    data = []
    for run in os.listdir(output_root):
        run_dir = os.path.join(output_root, run)
        if os.path.isdir(run_dir):
            r = {'dir': run}
            for name in param_grid_names + ['results']:
                json_file = os.path.join(run_dir, f'{name}.json')
                try:
                    with open(json_file, 'r') as fp:
                        d = json.load(fp)
                        d = {f'{name}__{k}': d[k] for k in d.keys()}
                        r.update(d)
                except (FileNotFoundError, KeyError) as e:
                    print(str(e))
            data.append(r)

    csv_file = os.path.join(output_root, 'results.csv')
    df = pd.DataFrame(data)
    df.to_csv(csv_file, index=False)


def check_model(data_dir, output_path):
    clear_session()
    model_file = os.path.join(output_path, 'model.h5')
    preprocess_params_file = os.path.join(output_path, 'preprocess_params.json')
    with open(preprocess_params_file, 'r') as fp:
        preprocess_params = json.load(fp)

    model = load_model(model_file, compile=True)

    def preprocess_func(u):
        return preprocess(u, **preprocess_params)

    x, y = get_xy(data_dir, 'valid', preprocess_func)
    y_hat = model.predict(x).reshape((-1,))

    loss = reduce_mean(binary_crossentropy(y_true=y, y_pred=y_hat)).numpy()
    acc = reduce_mean(binary_accuracy(y_true=y, y_pred=y_hat)).numpy()

    results_file = os.path.join(output_path, 'results.json')
    with open(results_file, 'r') as fp:
        results = json.load(fp)

    print(f'Model at {output_path}:')
    print(f'\t{"Computed":25s}: loss = {loss:.6f}, acc = {acc:.4%}')
    print(f'\t{"Results file":25s}: loss = {results["valid_loss"]:.6f}, acc = {results["valid_acc"]:.4%}')

    assert abs(loss - results['valid_loss']) < 1e-6, f'Computed loss does not match recorded loss'
    assert abs(acc - results['valid_acc']) < 1e-6, f'Computed accuracy does not match recorded accuracy'


def main():
    """
    Example of how to do a grid search. Each process gets its own parameter grid.
    Itertools is used to pull one parameter set (dictionary) from each of those.

    :return: None
    """
    # paths
    data_dir = '/Users/sydne/CS3537/SquashProblem4/P4data'
    output_root = 'problem_4_output14'

    # dictionary of parameter grids, one for each process
    param_grids = {
        'preprocess_params': ParameterGrid({
            'target_size': [(150, 150)],
        }),
        'augmentation_params': ParameterGrid({
            'rescale': [None, 1. / 255],
            'horizontal_flip': [True],
            'rotation_range': [40],
            'width_shift_range': [0.2],
            'height_shift_range': [0.2],
            'shear_range': [0.2],
            'zoom_range': [0.2],
            # 'featurewise_center': [True],
            # 'featurewise_std_normalization': [True]
        }),
        'model_params': ParameterGrid({
            'optimizer': ['nadam'],
        }),
        'generator_params': ParameterGrid({
            'batch_size': [16, 32],
        }),
        'early_stopping_params': ParameterGrid({
            'patience': [5],
        }),
        'fit_params': ParameterGrid({
            'epochs': [10],
        }),
    }

    # create list of names and corresponding parameter grids for use with itertools.product
    param_grid_names = list(param_grids.keys())
    param_grid_list = [param_grids[k] for k in param_grid_names]

    for params in itertools.product(*param_grid_list):
        # store parameters in dictionary, one item per process
        params = {k: v for k, v in zip(param_grid_names, params)}
        print('params:', params)

        # setup output directory
        date = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
        output_path = os.path.join(output_root, f'{date}')
        os.makedirs(output_path, exist_ok=True)

        # make the single argument preprocess function
        def preprocess_func(x):
            return preprocess(x, **params['preprocess_params'])

        # load data and preprocess it with preprocess parameters
        x_train, y_train = get_xy(data_dir, 'train', preprocess_func)
        x_valid, y_valid = get_xy(data_dir, 'valid', preprocess_func)

        # create training data generator with generator parameters
        train_generator = ImageDataGenerator(**params['augmentation_params']).flow(x_train, y_train,
                                                                                   **params['generator_params'])

        # build model with model parameters
        model = build_fn(input_shape=x_valid.shape[1:], output_activation='sigmoid', metrics=['accuracy'],
                         loss='binary_crossentropy', **params['model_params'])

        # setup early stopping with early stopping parameters
        early_stopping = EarlyStopping(monitor='val_loss', verbose=1, **params['early_stopping_params'])
        # setup model checkpointing
        model_checkpoint = ModelCheckpoint(
            filepath=os.path.join(output_path, 'model.h5'),
            save_weights_only=False, save_freq='epoch',
            save_best_only=True, monitor='val_loss', verbose=1)
        callbacks = [early_stopping, model_checkpoint]

        # update fit params based on batch size
        params['fit_params'].update(dict(
            steps_per_epoch=len(x_train) // params['generator_params']['batch_size'],
        ))

        # save parameters to output path
        for k, v in params.items():
            with open(os.path.join(output_path, f'{k}.json'), 'w') as fp:
                json.dump(v, fp)

        # save preprocess function
        with open(os.path.join(output_path, 'preprocess.py'), 'w') as fp:
            fp.write(inspect.getsource(preprocess))

        # save build_fn function
        with open(os.path.join(output_path, 'build_fn.py'), 'w') as fp:
            fp.write(inspect.getsource(build_fn))

        # train model
        model.fit(train_generator, validation_data=(x_valid, y_valid), callbacks=callbacks, verbose=1,
                  **params['fit_params'])

        # get and save results
        results = {}
        for target in ['train', 'valid']:
            results.update(analyze_results(output_path, data_dir, target, preprocess_func))
        with open(os.path.join(output_path, 'results.json'), 'w') as fp:
            json.dump(results, fp)

        # save a file with name that shows validation performance (for convenience)
        with open(os.path.join(output_path, f'{results["valid_loss"]}_{results["valid_acc"]}.out'), 'w') as fp:
            pass

        check_model(data_dir, output_path)

    # assemble results from all runs into one CSV file in output root.
    assemble_results(output_root, param_grid_names)


if __name__ == '__main__':
    main()
