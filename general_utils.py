import pandas as pd
import numpy as np
import math
import os


def global_extrema(dir_name):
    """Return the global maxima and minima of Ir191Di___191Ir_DNA1 and
       Event_length in all the files under the given directory."""
    # The following helper function returns the relavant maxima and minima
    # for a specific file
    def local_extrema(filename):
        data = pd.read_csv(os.path.join(dir_name, filename))
        ir191 = data["Ir191Di___191Ir_DNA1"]
        el = data["Event_length"]
        return (ir191.max(), ir191.min(), el.max(), el.min())
    ir191_max, el_max = -math.inf, -math.inf
    ir191_min, el_min = math.inf, math.inf
    # find the global maxima and minima for the two parameters across
    # all the files in the directory
    for file in os.listdir(dir_name):
        ir191_max_l, ir191_min_l, el_max_l, el_min_l = local_extrema(file)
        ir191_max = max(ir191_max, ir191_max_l)
        ir191_min = min(ir191_min, ir191_min_l)
        el_max = max(el_max, el_max_l)
        el_min = min(el_min, el_min_l)
    return (ir191_max, ir191_min, el_max, el_min)


# NOTE: the following conversion depends on the result of the function 'global_extrema' 
def axes_convert(df_numpy):
    """Convert the axes according to the following rules:
       ir191: int(ir191 * 10000 // 622)
       el: int(el - 10)
       The input is a numpy array of shape (cell_numbers, 3). The output is a numpy
       array of the same shape as the input."""
    for idx in range(df_numpy.shape[0]):
        ir191, el, gate1 = df_numpy[idx]
        ir191 = ir191 * 10000 // 622
        el = el - 10
        df_numpy[idx] = np.array((ir191, el, gate1))
    return df_numpy


def convert_to_image(dir_name, filename, mode="visual"):
    """Read a data file and output a shrinked greyscale image for the dataset.
       The output image is a numpy array of shape (166, 166, 1).
       modes:
       visual (default) - to visualise the gate
       train - to produce an image for training
       label - to produce a label for the dataset
       """
    data = pd.read_csv(os.path.join(dir_name, filename))[["Ir191Di___191Ir_DNA1", "Event_length", "gate1_ir"]]
    data = axes_convert(data.to_numpy())
    
    if mode == "visual" or mode == "train":
        image = np.zeros((166, 166, 1))
        if mode == "visual":
            for cell in data:
                ir191, el, gate1 = cell
                ir191, el, gate1 = int(ir191), int(el), int(gate1)  # need to make sure these variables are integers
                if gate1 == 0:
                    image[ir191, el, 0] = 128
                elif gate1 == 1:
                    image[ir191, el, 0] = 255
        elif mode == "train":
            for cell in data:
                ir191, el, gate1 = cell
                ir191, el, gate1 = int(ir191), int(el), int(gate1)  # need to make sure these variables are integers
                image[ir191, el, 0] += 1
        return image
    elif mode == "label":
        label = np.zeros((166, 166))
        for cell in data:
            ir191, el, gate1 = cell
            ir191, el, gate1 = int(ir191), int(el), int(gate1)  # need to make sure these variables are integers
            if gate1 == 1:
                label[ir191, el] = 1
        return label
        