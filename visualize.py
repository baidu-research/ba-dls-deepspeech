r"""
Use this script to visualize the output of a trained speech-model.
Usage: python visualize.py /path/to/audio /path/to/training/json.json \
            /path/to/model
"""

from __future__ import absolute_import, division, print_function
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from data_generator import DataGenerator
from model import compile_output_fn
from utils import argmax_decode, load_model


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def visualize(model, test_file, train_desc_file):
    """ Get the prediction using the model, and visualize softmax outputs
    Params:
        model (keras.models.Model): Trained speech model
        test_file (str): Path to an audio clip
        train_desc_file(str): Path to the training file used to train this
                              model
    """
    datagen = DataGenerator()
    datagen.load_train_data(train_desc_file)
    datagen.fit_train(100)

    print ("Compiling test function...")
    test_fn = compile_output_fn(model)

    inputs = [datagen.featurize(test_file)]

    prediction = np.squeeze(test_fn([inputs, True]))
    softmax_file = "softmax.npy".format(test_file)
    softmax_img_file = "softmax.png".format(test_file)
    print ("Prediction: {}"
           .format(argmax_decode(prediction)))
    print ("Saving network output to: {}".format(softmax_file))
    print ("As image: {}".format(softmax_img_file))
    np.save(softmax_file, prediction)
    sm = softmax(prediction.T)
    sm = np.vstack((sm[0], sm[2], sm[3:][::-1]))
    fig, ax = plt.subplots()
    ax.pcolor(sm, cmap=plt.cm.Greys_r)
    column_labels = [chr(i) for i in range(97, 97 + 26)] + ['space', 'blank']
    ax.set_yticks(np.arange(sm.shape[0]) + 0.5, minor=False)
    ax.set_yticklabels(column_labels[::-1], minor=False)
    plt.savefig(softmax_img_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('test_file', type=str,
                        help='Path to an audio file')
    parser.add_argument('train_desc_file', type=str,
                        help='Path to the training JSON-line file. This will '
                             'be used to extract feature means/variance')
    parser.add_argument('load_dir', type=str,
                        help='Directory where a trained model is stored.')
    parser.add_argument('--weights_file', type=str, default=None,
                        help='Path to a model weights file')
    args = parser.parse_args()

    print ("Loading model")
    model = load_model(args.load_dir, args.weights_file)
    visualize(model, args.test_file, args.train_desc_file)


if __name__ == '__main__':
    main()
