"""
Defines a class that is used to featurize audio clips, and provide
them to the network for training or testing.
"""

from __future__ import absolute_import, division, print_function
from functools import reduce

import json
import logging
import numpy as np
import random

from concurrent.futures import ThreadPoolExecutor, wait

from utils import calc_feat_dim, spectrogram_from_file, text_to_int_sequence

RNG_SEED = 123
logger = logging.getLogger(__name__)


class DataGenerator(object):
    def __init__(self, step=10, window=20, max_freq=8000, desc_file=None):
        """
        Params:
            step (int): Step size in milliseconds between windows
            window (int): FFT window size in milliseconds
            max_freq (int): Only FFT bins corresponding to frequencies between
                [0, max_freq] are returned
            desc_file (str, optional): Path to a JSON-line file that contains
                labels and paths to the audio files. If this is None, then
                load metadata right away
        """
        self.feat_dim = calc_feat_dim(window, max_freq)
        self.feats_mean = np.zeros((self.feat_dim,))
        self.feats_std = np.ones((self.feat_dim,))
        self.rng = random.Random(RNG_SEED)
        if desc_file is not None:
            self.load_metadata_from_desc_file(desc_file)
        self.step = step
        self.window = window
        self.max_freq = max_freq

    def featurize(self, audio_clip):
        """ For a given audio clip, calculate the log of its Fourier Transform
        Params:
            audio_clip(str): Path to the audio clip
        """
        return spectrogram_from_file(
            audio_clip, step=self.step, window=self.window,
            max_freq=self.max_freq)

    def load_metadata_from_desc_file(self, desc_file, partition='train',
                                     max_duration=10.0,):
        """ Read metadata from the description file
            (possibly takes long, depending on the filesize)
        Params:
            desc_file (str):  Path to a JSON-line file that contains labels and
                paths to the audio files
            partition (str): One of 'train', 'validation' or 'test'
            max_duration (float): In seconds, the maximum duration of
                utterances to train or test on
        """
        logger.info('Reading description file: {} for partition: {}'
                    .format(desc_file, partition))
        audio_paths, durations, texts = [], [], []
        with open(desc_file) as json_line_file:
            for line_num, json_line in enumerate(json_line_file):
                try:
                    spec = json.loads(json_line)
                    if float(spec['duration']) > max_duration:
                        continue
                    audio_paths.append(spec['key'])
                    durations.append(float(spec['duration']))
                    texts.append(spec['text'])
                except Exception as e:
                    # Change to (KeyError, ValueError) or
                    # (KeyError,json.decoder.JSONDecodeError), depending on
                    # json module version
                    logger.warn('Error reading line #{}: {}'
                                .format(line_num, json_line))
                    logger.warn(str(e))

        if partition == 'train':
            self.train_audio_paths = audio_paths
            self.train_durations = durations
            self.train_texts = texts
        elif partition == 'validation':
            self.val_audio_paths = audio_paths
            self.val_durations = durations
            self.val_texts = texts
        elif partition == 'test':
            self.test_audio_paths = audio_paths
            self.test_durations = durations
            self.test_texts = texts
        else:
            raise Exception("Invalid partition to load metadata. "
                            "Must be train/validation/test")

    def load_train_data(self, desc_file):
        self.load_metadata_from_desc_file(desc_file, 'train')

    def load_test_data(self, desc_file):
        self.load_metadata_from_desc_file(desc_file, 'test')

    def load_validation_data(self, desc_file):
        self.load_metadata_from_desc_file(desc_file, 'validation')

    @staticmethod
    def sort_by_duration(durations, audio_paths, texts):
        return zip(*sorted(zip(durations, audio_paths, texts)))

    def normalize(self, feature, eps=1e-14):
        return (feature - self.feats_mean) / (self.feats_std + eps)

    def prepare_minibatch(self, audio_paths, texts):
        """ Featurize a minibatch of audio, zero pad them and return a dictionary
        Params:
            audio_paths (list(str)): List of paths to audio files
            texts (list(str)): List of texts corresponding to the audio files
        Returns:
            dict: See below for contents
        """
        assert len(audio_paths) == len(texts),\
            "Inputs and outputs to the network must be of the same number"
        # Features is a list of (timesteps, feature_dim) arrays
        # Calculate the features for each audio clip, as the log of the
        # Fourier Transform of the audio
        features = [self.featurize(a) for a in audio_paths]
        input_lengths = [f.shape[0] for f in features]
        max_length = max(input_lengths)
        feature_dim = features[0].shape[1]
        mb_size = len(features)
        # Pad all the inputs so that they are all the same length
        x = np.zeros((mb_size, max_length, feature_dim))
        y = []
        label_lengths = []
        for i in range(mb_size):
            feat = features[i]
            feat = self.normalize(feat)  # Center using means and std
            x[i, :feat.shape[0], :] = feat
            label = text_to_int_sequence(texts[i])
            y.append(label)
            label_lengths.append(len(label))
        # Flatten labels to comply with warp-CTC signature
        y = reduce(lambda i, j: i + j, y)
        return {
            'x': x,  # (0-padded features of shape(mb_size,timesteps,feat_dim)
            'y': y,  # list(int) Flattened labels (integer sequences)
            'texts': texts,  # list(str) Original texts
            'input_lengths': input_lengths,  # list(int) Length of each input
            'label_lengths': label_lengths  # list(int) Length of each label
        }

    def iterate(self, audio_paths, texts, minibatch_size,
                max_iters=None):
        if max_iters is not None:
            k_iters = max_iters
        else:
            k_iters = int(np.ceil(len(audio_paths) / minibatch_size))
        logger.info("Iters: {}".format(k_iters))
        pool = ThreadPoolExecutor(1)  # Run a single I/O thread in parallel
        future = pool.submit(self.prepare_minibatch,
                             audio_paths[:minibatch_size],
                             texts[:minibatch_size])
        start = minibatch_size
        for i in range(k_iters - 1):
            wait([future])
            minibatch = future.result()
            # While the current minibatch is being consumed, prepare the next
            future = pool.submit(self.prepare_minibatch,
                                 audio_paths[start: start + minibatch_size],
                                 texts[start: start + minibatch_size])
            yield minibatch
            start += minibatch_size
        # Wait on the last minibatch
        wait([future])
        minibatch = future.result()
        yield minibatch

    def iterate_train(self, minibatch_size=16, sort_by_duration=False,
                      shuffle=True):
        if sort_by_duration and shuffle:
            shuffle = False
            logger.warn("Both sort_by_duration and shuffle were set to True. "
                        "Setting shuffle to False")
        durations, audio_paths, texts = (self.train_durations,
                                         self.train_audio_paths,
                                         self.train_texts)
        if shuffle:
            temp = zip(durations, audio_paths, texts)
            self.rng.shuffle(temp)
            durations, audio_paths, texts = zip(*temp)
        if sort_by_duration:
            durations, audio_paths, texts =\
                DataGenerator.sort_by_duration(durations, audio_paths, texts)
        return self.iterate(audio_paths, texts, minibatch_size)

    def iterate_test(self, minibatch_size=16):
        return self.iterate(self.test_audio_paths, self.test_texts,
                            minibatch_size)

    def iterate_validation(self, minibatch_size=16):
        return self.iterate(self.val_audio_paths, self.val_texts,
                            minibatch_size)

    def fit_train(self, k_samples=100):
        """ Estimate the mean and std of the features from the training set
        Params:
            k_samples (int): Use this number of samples for estimation
        """
        k_samples = min(k_samples, len(self.train_audio_paths))
        samples = self.rng.sample(self.train_audio_paths, k_samples)
        feats = [self.featurize(s) for s in samples]
        feats = np.vstack(feats)
        self.feats_mean = np.mean(feats, axis=0)
        self.feats_std = np.std(feats, axis=0)
