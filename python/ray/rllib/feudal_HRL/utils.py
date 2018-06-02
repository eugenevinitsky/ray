from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

def flatten(weights, start=0, stop=2):
    """This methods reshapes all values in a dictionary.

    The indices from start to stop will be flattened into a single index.

    Args:
        weights: A dictionary mapping keys to numpy arrays.
        start: The starting index.
        stop: The ending index.
    """
    for key, val in weights.items():
        new_shape = val.shape[0:start] + (-1,) + val.shape[stop:]
        weights[key] = val.reshape(new_shape)
    return weights


def concatenate(weights_list):
    keys = weights_list[0].keys()
    result = {}
    for key in keys:
        result[key] = np.concatenate([l[key] for l in weights_list])
    return result


def shuffle(trajectory):
    permutation = np.random.permutation(trajectory["actions"].shape[0])
    for key, val in trajectory.items():
        trajectory[key] = val[permutation]
    return trajectory

def log_histogram(writer, tag, values, step, bins=1000):
        """Logs the histogram of a list/vector of values."""
        # Convert to a numpy array
        values = np.array(values)

        # Create histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill fields of histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values ** 2))

        # Requires equal number as bins, where the first goes from -DBL_MAX to bin_edges[1]
        # See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto#L30
        # Thus, we drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        writer.add_summary(summary, step)