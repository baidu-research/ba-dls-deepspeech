import glob
import logging
import os
import numpy as np
import re
import soundfile
from keras.models import model_from_json
from numpy.lib.stride_tricks import as_strided

from char_map import char_map, index_map

logger = logging.getLogger(__name__)


def calc_feat_dim(window, max_freq):
    return int(0.001 * window * max_freq) + 1


def conv_output_length(input_length, filter_size, border_mode, stride,
                       dilation=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    return (output_length + stride - 1) // stride


def spectrogram(samples, fft_length=256, sample_rate=2, hop_length=128):
    """
    Compute the spectrogram for a real signal.
    The parameters follow the naming convention of
    matplotlib.mlab.specgram

    Args:
        samples (1D array): input audio signal
        fft_length (int): number of elements in fft window
        sample_rate (scalar): sample rate
        hop_length (int): hop length (relative offset between neighboring
            fft windows).

    Returns:
        x (2D array): spectrogram [frequency x time]
        freq (1D array): frequency of each row in x

    Note:
        This is a truncating computation e.g. if fft_length=10,
        hop_length=5 and the signal has 23 elements, then the
        last 3 elements will be truncated.
    """
    assert not np.iscomplexobj(samples), "Must not pass in complex numbers"

    window = np.hanning(fft_length)[:, None]
    window_norm = np.sum(window**2)

    # The scaling below follows the convention of
    # matplotlib.mlab.specgram which is the same as
    # matlabs specgram.
    scale = window_norm * sample_rate

    trunc = (len(samples) - fft_length) % hop_length
    x = samples[:len(samples) - trunc]

    # "stride trick" reshape to include overlap
    nshape = (fft_length, (len(x) - fft_length) // hop_length + 1)
    nstrides = (x.strides[0], x.strides[0] * hop_length)
    x = as_strided(x, shape=nshape, strides=nstrides)

    # window stride sanity check
    assert np.all(x[:, 1] == samples[hop_length:(hop_length + fft_length)])

    # broadcast window, compute fft over columns and square mod
    x = np.fft.rfft(x * window, axis=0)
    x = np.absolute(x)**2

    # scale, 2.0 for everything except dc and fft_length/2
    x[1:-1, :] *= (2.0 / scale)
    x[(0, -1), :] /= scale

    freqs = float(sample_rate) / fft_length * np.arange(x.shape[0])

    return x, freqs


def spectrogram_from_file(filename, step=10, window=20, max_freq=None,
                          eps=1e-14):
    """ Calculate the log of linear spectrogram from FFT energy
    Params:
        filename (str): Path to the audio file
        step (int): Step size in milliseconds between windows
        window (int): FFT window size in milliseconds
        max_freq (int): Only FFT bins corresponding to frequencies between
            [0, max_freq] are returned
        eps (float): Small value to ensure numerical stability (for ln(x))
    """
    with soundfile.SoundFile(filename) as sound_file:
        audio = sound_file.read(dtype='float32')
        sample_rate = sound_file.samplerate
        if audio.ndim >= 2:
            audio = np.mean(audio, 1)
        if max_freq is None:
            max_freq = sample_rate / 2
        if max_freq > sample_rate / 2:
            raise ValueError("max_freq must not be greater than half of "
                             " sample rate")
        if step > window:
            raise ValueError("step size must not be greater than window size")
        hop_length = int(0.001 * step * sample_rate)
        fft_length = int(0.001 * window * sample_rate)
        pxx, freqs = spectrogram(
            audio, fft_length=fft_length, sample_rate=sample_rate,
            hop_length=hop_length)
        ind = np.where(freqs <= max_freq)[0][-1] + 1
    return np.transpose(np.log(pxx[:ind, :] + eps))


def save_model(save_dir, model, train_costs, validation_costs, index=None):
    """ Save the model and costs into a directory
    Params:
        save_dir (str): Directory used to store the model
        model (keras.models.Model)
        train_costs (list(float))
        validation_costs (list(float))
        index (int): If this is provided, add this index as a suffix to
            the weights (useful for checkpointing during training)
    """
    logger.info("Checkpointing model to: {}".format(save_dir))
    model_config_path = os.path.join(save_dir, 'model_config.json')
    with open(model_config_path, 'w') as model_config_file:
        model_json = model.to_json()
        model_config_file.write(model_json)
    if index is None:
        weights_format = 'model_weights.h5'
    else:
        weights_format = 'model_{}_weights.h5'.format(index)
    model_weights_file = os.path.join(save_dir, weights_format)
    model.save_weights(model_weights_file, overwrite=True)
    np.savez(os.path.join(save_dir, 'costs.npz'), train=train_costs,
             validation=validation_costs)


def load_model(load_dir, weights_file=None):
    """ Load a model and its weights from a directory
    Params:
        load_dir (str): Path the model directory
        weights_file (str): If this is not passed in, try to load the latest
            model_*weights.h5 file in the directory
    Returns:
        model (keras.models.Model)
    """
    def atoi(text):
        return int(text) if text.isdigit() else text

    def natural_keys(text):
        # From http://stackoverflow.com/questions/5967500
        return [atoi(c) for c in re.split('(\d+)', text)]

    model_config_file = os.path.join(load_dir, 'model_config.json')
    model_config = open(model_config_file).read()
    model = model_from_json(model_config)

    if weights_file is None:
        # This will find all files of name model_*weights.h5
        # We try to use the latest one saved
        weights_files = glob.glob(os.path.join(load_dir, 'model_*weights.h5'))
        weights_files.sort(key=natural_keys)
        model_weights_file = weights_files[-1]  # Use the latest model
    else:
        model_weights_file = weights_file
    model.load_weights(model_weights_file)
    return model


def argmax_decode(prediction):
    """ Decode a prediction using the highest probable character at each
        timestep. Then, simply convert the integer sequence to text
    Params:
        prediction (np.array): timestep * num_characters
    """
    int_sequence = []
    for timestep in prediction:
        int_sequence.append(np.argmax(timestep))
    tokens = []
    c_prev = -1
    for c in int_sequence:
        if c == c_prev:
            continue
        if c != 0:  # Blank
            tokens.append(c)
        c_prev = c

    text = ''.join([index_map[i] for i in tokens])
    return text


def text_to_int_sequence(text):
    """ Use a character map and convert text to an integer sequence """
    int_sequence = []
    for c in text:
        if c == ' ':
            ch = char_map['<SPACE>']
        else:
            ch = char_map[c]
        int_sequence.append(ch)
    return int_sequence


def configure_logging(console_log_level=logging.INFO,
                      console_log_format=None,
                      file_log_path=None,
                      file_log_level=logging.INFO,
                      file_log_format=None,
                      clear_handlers=False):
    """Setup logging.

    This configures either a console handler, a file handler, or both and
    adds them to the root logger.

    Args:
        console_log_level (logging level): logging level for console logger
        console_log_format (str): log format string for console logger
        file_log_path (str): full filepath for file logger output
        file_log_level (logging level): logging level for file logger
        file_log_format (str): log format string for file logger
        clear_handlers (bool): clear existing handlers from the root logger

    Note:
        A logging level of `None` will disable the handler.
    """
    if file_log_format is None:
        file_log_format = \
            '%(asctime)s %(levelname)-7s (%(name)s) %(message)s'

    if console_log_format is None:
        console_log_format = \
            '%(asctime)s %(levelname)-7s (%(name)s) %(message)s'

    # configure root logger level
    root_logger = logging.getLogger()
    root_level = root_logger.level
    if console_log_level is not None:
        root_level = min(console_log_level, root_level)
    if file_log_level is not None:
        root_level = min(file_log_level, root_level)
    root_logger.setLevel(root_level)

    # clear existing handlers
    if clear_handlers and len(root_logger.handlers) > 0:
        print("Clearing {} handlers from root logger."
              .format(len(root_logger.handlers)))
        for handler in root_logger.handlers:
            root_logger.removeHandler(handler)

    # file logger
    if file_log_path is not None and file_log_level is not None:
        log_dir = os.path.dirname(os.path.abspath(file_log_path))
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        file_handler = logging.FileHandler(file_log_path)
        file_handler.setLevel(file_log_level)
        file_handler.setFormatter(logging.Formatter(file_log_format))
        root_logger.addHandler(file_handler)

    # console logger
    if console_log_level is not None:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_log_level)
        console_handler.setFormatter(logging.Formatter(console_log_format))
        root_logger.addHandler(console_handler)
