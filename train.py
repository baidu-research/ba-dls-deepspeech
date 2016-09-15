from __future__ import absolute_import, division, print_function

import argparse
import logging
import os

from data_generator import DataGenerator
from model import compile_gru_model, compile_train_fn, compile_val_fn
from utils import configure_logging, save_model

logger = logging.getLogger(__name__)


def validation(model, val_fn, datagen, mb_size=16):
    """ Validation routine for speech-models
    Params:
        model (keras.model): Constructed keras model
        val_fn (theano.function): A theano function that calculates the cost
            over a validation set
        datagen (DataGenerator)
        mb_size (int): Size of each minibatch
    Returns:
        val_cost (float): Average validation cost over the whole validation set
    """
    avg_cost = 0.0
    i = 0
    for batch in datagen.iterate_validation(mb_size):
        inputs = batch['x']
        labels = batch['y']
        input_lengths = batch['input_lengths']
        label_lengths = batch['label_lengths']
        output_lengths = [model.conv_output_length(l)
                          for l in input_lengths]
        _, ctc_cost = val_fn([inputs, output_lengths, labels,
                              label_lengths, False])
        avg_cost += ctc_cost
        i += 1
    return avg_cost / i


def train(model, train_fn, val_fn, datagen, save_dir, epochs=10, mb_size=16):
    """ Main training routine for speech-models
    Params:
        model (keras.model): Constructed keras model
        train_fn (theano.function): A theano function that takes in acoustic
            inputs and updates the model
        val_fn (theano.function): A theano function that calculates the cost
            over a validation set
        datagen (DataGenerator)
        save_dir (str): Path where model and costs are saved
        epochs (int): Total epochs to continue training
        mb_size (int): Size of each minibatch
    """
    train_costs, val_costs = [], []
    iters = 0
    for e in range(epochs):
        sortagrad = e == 0
        for i, batch in \
                enumerate(datagen.iterate_train(mb_size,
                                                sort_by_duration=sortagrad)):
            inputs = batch['x']
            labels = batch['y']
            input_lengths = batch['input_lengths']
            label_lengths = batch['label_lengths']
            output_lengths = [model.conv_output_length(l)
                              for l in input_lengths]
            _, ctc_cost = train_fn([inputs, output_lengths, labels,
                                    label_lengths, True])
            train_costs.append(ctc_cost)
            if i % 10 == 0:
                logger.info("Epoch: {}, Iteration: {}, Loss: {}"
                            .format(e, i, ctc_cost, input_lengths))
            iters += 1
            if iters % 500 == 0:
                val_cost = validation(model, val_fn, datagen, mb_size)
                val_costs.append(val_cost)
                save_model(save_dir, model, train_costs, val_costs)
        if iters % 500 != 0:
            # End of an epoch. Check validation cost and save costs
            val_cost = validation(model, val_fn, datagen, mb_size)
            val_costs.append(val_cost)
            save_model(save_dir, model, train_costs, val_costs)


def main(desc_file, epochs, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # Configure logging
    configure_logging(file_log_path=os.path.join(save_dir, 'train_log.txt'))

    # Prepare the data generator
    datagen = DataGenerator()
    datagen.load_metadata_from_desc_file(args.desc_file)
    datagen.fit_train(6)

    # Compile a model
    model = compile_gru_model(recur_layers=3, nodes=500, batch_norm=True)

    # Compile the CTC training function
    train_fn = compile_train_fn(model)

    # Compile the validation function
    val_fn = compile_val_fn(model)

    # Train the model
    train(model, train_fn, val_fn, datagen, save_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('desc_file', type=str,
                        help='Path to a JSON-line file that contains '
                             'labels and paths to the audio files. ')
    parser.add_argument('save_dir', type=str,
                        help='Directory to store the model. This will be '
                             'created if it doesn\'t already exist')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs to train the model')
    args = parser.parse_args()

    main(args.desc_file, args.epochs, args.save_dir)
