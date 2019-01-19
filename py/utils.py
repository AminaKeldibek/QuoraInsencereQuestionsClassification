import os
import errno
import numpy as np


def create_dir(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise


def binarize(x, threshold):
    x_out = np.zeros_like(x)
    x_out[x > threshold] = 1
    x_out[x <= threshold] = 0

    return (x_out)
