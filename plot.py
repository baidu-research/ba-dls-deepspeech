from __future__ import division
from __future__ import print_function
import argparse
import matplotlib
import numpy as np
import os
# import random
matplotlib.use('Agg')  # This must be called before importing pyplot
import matplotlib.pyplot as plt


# Better Color Scheme
COLORS_RGB = [
    (228, 26, 28), (55, 126, 184), (77, 175, 74),
    (152, 78, 163), (255, 127, 0), (255, 255, 51),
    (166, 86, 40), (247, 129, 191), (153, 153, 153)
]

# Worse Color Scheme - use random.shuffle() below if using this
# COLORS_RGB = [
#    (31, 119, 180), (255, 127, 14), (44, 160, 44), (219, 75, 75),
#    (174, 199, 232), (255, 187, 120), (152, 223, 138), (255, 152, 150),
#    (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
#    (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
#    (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)
# ]

# Scale the RGB values to the [0, 1] range, which is the format
# matplotlib accepts.
colors = [(r / 255, g / 255, b / 255) for r, g, b in COLORS_RGB]
# random.shuffle(colors) # use only if using worse color scheme


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dirs', nargs='+', required=True)
    parser.add_argument('-s', '--save_file', type=str, required=True)
    return parser.parse_args()


def graph(dirs, save_file, average_window=100):
    fig, ax = plt.subplots()
    ax.set_xlabel('Iters')
    ax.set_ylabel('Loss')
    average_filter = np.ones(average_window) / float(average_window)

    for i, d in enumerate(dirs):
        name = os.path.basename(os.path.abspath(d))
        color = colors[i % len(colors)]
        costs = np.load(os.path.join(d, 'costs.npz'))
        train_costs = costs['train']
        valid_costs = costs['validation']
        if train_costs.ndim == 1:
            train_costs = np.convolve(train_costs, average_filter,
                                      mode='valid')
        if len(valid_costs.shape) > 1:
            valid_costs = valid_costs[:, 0]  # Remove accuracy or other metrics
        ax.plot(train_costs, color=color, label=name + '_train', lw=1.5)
        ax.plot(valid_costs, '--', color=color, label=name + '_valid')
    ax.grid(True)
    ax.legend(loc='best')
    plt.savefig(save_file)


if __name__ == '__main__':
    args = parse_args()
    graph(args.dirs, args.save_file)
