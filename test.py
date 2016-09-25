"""
Test a trained speech model over a dataset
"""

from __future__ import absolute_import, division, print_function
import argparse
import numpy as np

from data_generator import DataGenerator
from model import compile_test_fn
from utils import argmax_decode, conv_output_length, load_model


def test(model, test_fn, datagen, mb_size=16, conv_context=11,
         conv_border_mode='valid', conv_stride=2):
    """ Testing routine for speech-models
    Params:
        model (keras.model): Constructed keras model
        test_fn (theano.function): A theano function that calculates the cost
            over a test set
        datagen (DataGenerator)
        mb_size (int): Size of each minibatch
        conv_context (int): Convolution context
        conv_border_mode (str): Convolution border mode
        conv_stride (int): Convolution stride
    Returns:
        test_cost (float): Average test cost over the whole test set
    """
    avg_cost = 0.0
    i = 0
    for batch in datagen.iterate_test(mb_size):
        inputs = batch['x']
        labels = batch['y']
        input_lengths = batch['input_lengths']
        label_lengths = batch['label_lengths']
        ground_truth = batch['texts']
        # Due to convolution, the number of timesteps of the output
        # is different from the input length. Calculate the resulting
        # timesteps
        output_lengths = [conv_output_length(l, conv_context,
                                             conv_border_mode, conv_stride)
                          for l in input_lengths]
        predictions, ctc_cost = test_fn([inputs, output_lengths, labels,
                                        label_lengths, True])
        predictions = np.swapaxes(predictions, 0, 1)
        for i, prediction in enumerate(predictions):
            print ("Truth: {}, Prediction: {}"
                   .format(ground_truth[i], argmax_decode(prediction)))
        avg_cost += ctc_cost
        i += 1
    return avg_cost / i


def main(test_desc_file, train_desc_file, load_dir):
    # Prepare the data generator
    datagen = DataGenerator()
    # Load the JSON file that contains the dataset
    datagen.load_test_data(test_desc_file)
    datagen.load_train_data(train_desc_file)
    # Use a few samples from the dataset, to calculate the means and variance
    # of the features, so that we can center our inputs to the network
    datagen.fit_train(100)

    # Compile a Recurrent Network with 1 1D convolution layer, GRU units
    # and 1 fully connected layer
    model = load_model(load_dir)

    # Compile the testing function
    test_fn = compile_test_fn(model)

    # Test the model
    test_loss = test(model, test_fn, datagen)
    print ("Test loss: {}".format(test_loss))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('test_desc_file', type=str,
                        help='Path to a JSON-line file that contains '
                             'test labels and paths to the audio files. ')
    parser.add_argument('train_desc_file', type=str,
                        help='Path to the training JSON-line file. This will '
                             'be used to extract feature means/variance')
    parser.add_argument('load_dir', type=str,
                        help='Directory where a trained model is stored.')
    args = parser.parse_args()
    main(args.test_desc_file, args.train_desc_file, args.load_dir)
