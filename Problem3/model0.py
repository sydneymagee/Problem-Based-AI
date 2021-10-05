import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow
import tensorflow_core
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Lambda, Input, Dropout, LeakyReLU
from tensorflow_core.python.keras.layers.preprocessing.normalization import Normalization
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.losses import mean_squared_error, kullback_leibler_divergence


def get_xy(npz_file, endpoint='pitch'):
    """
    Gets the data matrix, X, and target, y for the desired 'endpoint'.
    :param npz_file: The path to the npz file
    :param endpoint: 'pitches', 'timbre', or 'loudness'
    :return:
    """
    with np.load(npz_file) as data:
        x = data['x']
        key = f'y_{endpoint}'
        if key not in data:
            raise ValueError(f'Unknown endpoint {endpoint}')
        y = data[key]
        if y.ndim == 1:
            y = data[key].reshape((-1, 1))
    return x, y


def setup_model_checkpoints(output_path, save_freq):
    """
    Setup model checkpoints using the save path and frequency.
    :param output_path: The directory to store the checkpoints in
    :param save_freq: The frequency with which to save them "epoch" means each epoch
                      See ModelCheckpoint documentation.
    :return: a ModelCheckpoint
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    model_checkpoint = ModelCheckpoint(
        os.path.join(output_path, 'model.{epoch:05d}_{val_loss:f}.h5'),
        save_weights_only=False,
        save_freq=save_freq,
        save_best_only=True,
        monitor='val_loss',
        verbose=1
    )
    return model_checkpoint


def visualize(model, x, y_true, name='', output_path=''):
    """
    Create a joint distribution plot that shows relationship between
    model estimates and true values.

    :param model: A trained model
    :param x: The data matrix, X
    :param y_true: The target vector or matrix, y
    :param name: The name for the figure
    :param output_path: The output directory to save the PNG
    :return: None
    """
    png_file = os.path.join(output_path, f'visualize_{name}.png')
    y_pred = model.predict(x)

    # loss should be mean squared error unless pitch target
    metric = model.metrics[0]

    # check assumptions
    assert y_true.shape == y_pred.shape, f'Model output should have shape {y_true.shape} not {y_pred.shape}'
    if np.allclose(y_true.sum(axis=1), 1) and (y_true >= 0).all():
        # if each row of target sums to one and is nonnegative, we know it must be pitches.
        assert np.allclose(y_pred.sum(axis=1), 1), f'Model output should sum to one'
        assert (y_pred >= 0).all(), f'Model output should be nonnegative'
        metric = kullback_leibler_divergence

    loss = metric(y_true=y_true, y_pred=y_pred)

    # make joint plot
    jg = sns.jointplot(x=y_true.reshape((-1,)), y=y_pred.reshape((-1,)), kind='hist')
    jg.fig.suptitle(f'{name} (loss = {loss:.6f})')
    jg.set_axis_labels(xlabel='Actual', ylabel='Model')
    max_value = max(y_pred.max(), y_true.max())
    min_value = min(y_pred.min(), y_true.min())
    jg.ax_joint.plot([min_value, max_value], [min_value, max_value], color='k', linestyle='--')
    plt.tight_layout()
    plt.savefig(png_file)
    plt.close(jg.fig)


def get_best_model(output_path):
    """
    Parses the output_path to find the best model. Relies on the ModelCheckpoint
    saving a file name with the validation loss in it. If a model was saved with
    a Normalization layer, it's provided as a custom object.
    :param output_path: The directory to scan for H5 files
    :return: The best model compiled.
    """
    min_loss = float('inf')
    best_model_file = None
    for file_name in os.listdir(output_path):
        if file_name.endswith('.h5'):
            val_loss = float('.'.join(file_name.split('_')[1].split('.')[:-1]))
            if val_loss < min_loss:
                best_model_file = file_name
                min_loss = val_loss
    print(f'loading best model: {best_model_file}')
    model = keras.models.load_model(os.path.join(output_path, best_model_file), compile=True,
                                    custom_objects={'Normalization': Normalization})
    return model


def error_bits(error):
    """
    Return a lower bound on the number of bits to encode the errors based on Shannon's source coding theorem.
    :param error: Vector or list of errors (error = estimate - actual)
    :return: The lower bound number of bits to encode the errors
    """
    # round and cast to an integer, reshape as a vector
    error = np.round(error).astype(int).reshape((-1,))

    # shift so that the minimum value is zero
    error = error - error.min()

    # count how many occurrences of each discrete value
    p = np.bincount(error)

    # ignore zero counts
    p = p[p > 0]

    # convert counts into discrete probability distribution
    p = p / p.sum()

    # compute entropy (bits per codeword)
    entropy = -(p * np.log2(p)).sum()

    # minimum bits to encode all errors
    bits = int(np.ceil(entropy * len(error)))

    return bits


def get_improvement(model, x, y):
    y_hat = model.predict(x)
    error = y_hat - y
    mod_bits = 32 * model.count_params()
    orig_bits = error_bits(y)
    error_bit = error_bits(error)
    improvement = 1 - (mod_bits + error_bit) / orig_bits

    return improvement


def loudness():
    """
    An example applying linear regression to the loudness problem.
    :return: None
    """
    target = 'loudness'
    output_path = '/Users/sydne/CS3537/Spotify/spotify_loudness'

    X_train, y_train = get_xy('/Users/sydne/CS3537/Spotify/NPZ Files/spotify_train.npz', target)
    X_valid, y_valid = get_xy('/Users/sydne/CS3537/Spotify/NPZ Files/spotify_valid.npz', target)

    keras.backend.clear_session()
    # setup callbacks
    model_checkpoint = setup_model_checkpoints(output_path, save_freq='epoch')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

    # create linear model
    model = Sequential()
    model.add(Input(129))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dense(128))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dense(64))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dense(32))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dense(16))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dense(1, activation='linear'))

    print('Max: ', y_train.max())
    print('Starting Loudness: ', y_train[0])
    print('Ending Loudness: ', y_train[128])

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
    model.fit(X_train, y_train, verbose=1, validation_data=[X_valid, y_valid],
              callbacks=[model_checkpoint, early_stopping], epochs=5, batch_size=128)

    eval_train = model.evaluate(X_train, y_train, verbose=0)
    train_mse = eval_train[0]

    eval_val = model.evaluate(X_valid, y_valid, verbose=0)
    val_mse = eval_val[0]

    print(f'Parameters = {model.count_params():d}')
    print(f'train_mse = {train_mse:.4f}')
    print(f'val_mse = {val_mse:.4f}')

    # model.summary()
    visualize(model, x=X_train, y_true=y_train, name='Training', output_path=output_path)
    visualize(model, x=X_valid, y_true=y_valid, name='Validation', output_path=output_path)


loudness()
