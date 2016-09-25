"""
Plot training/validation curves for multiple models.
"""


from __future__ import division
from __future__ import print_function
import argparse
import matplotlib
import numpy as np
import os
matplotlib.use('Agg')  # This must be called before importing pyplot
import matplotlib.pyplot as plt


COLORS_RGB = [
    (228, 26, 28), (55, 126, 184), (77, 175, 74),
    (152, 78, 163), (255, 127, 0), (255, 255, 51),
    (166, 86, 40), (247, 129, 191), (153, 153, 153)
]

# Scale the RGB values to the [0, 1] range, which is the format
# matplotlib accepts.
colors = [(r / 255, g / 255, b / 255) for r, g, b in COLORS_RGB]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dirs', nargs='+', required=True,
                        help='Directories where the model and costs are saved')
    parser.add_argument('-s', '--save_file', type=str, required=True,
                        help='Filename of the output plot')
    return parser.parse_args()


def graph(dirs, save_file, average_window=100):
    """ Plot the training and validation costs over iterations
    Params:
        dirs (list(str)): Directories where the model and costs are saved
        save_file (str): Filename of the output plot
        average_window (int): Window size for smoothening the graphs
    """
    fig, ax = plt.subplots()
    ax.set_xlabel('Iters')
    ax.set_ylabel('Loss')
    average_filter = np.ones(average_window) / float(average_window)

    for i, d in enumerate(dirs):
        name = os.path.basename(os.path.abspath(d))
        color = colors[i % len(colors)]
        costs = np.load(os.path.join(d, 'costs.npz'))
        train_costs = costs['train']
        valid_costs = costs['validation'].tolist()
        iters = train_costs.shape[0]
        valid_range = [500 * (i + 1) for i in range(iters // 500)]
        if len(valid_range) != len(valid_costs):
            valid_range.append(iters)
        if train_costs.ndim == 1:
            train_costs = np.convolve(train_costs, average_filter,
                                      mode='valid')
        ax.plot(train_costs, color=color, label=name + '_train', lw=1.5)
        ax.plot(valid_range, valid_costs[:len(valid_range)],
                '-o', color=color, label=name + '_valid')
    ax.grid(True)
    ax.legend(loc='best')
    plt.savefig(save_file)


if __name__ == '__main__':
    args = parse_args()
    graph(args.dirs, args.save_file)
