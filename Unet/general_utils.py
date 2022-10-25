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


def convert_to_image(dir_name, filename, gate_name="gate1_ir", mode="visual"):
    """Convert the original csv files to images for visualising the gate, training, or GT labelling.
       modes:
       visual (default) - to visualise the gate
       train - to produce an image for training
       label - to produce a label for the dataset
       """
    data = pd.read_csv(os.path.join(dir_name, filename))[["Ir191Di___191Ir_DNA1", "Event_length", gate_name]]
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
                ir191, el, _ = cell
                ir191, el = int(ir191), int(el)  # need to make sure these variables are integers
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


# NOTE: this function is only used in FCN, not in Unet
def compute_weights(train_label_path):
    """Computes the ratio between the number of pixels labelled 0 and that of pixels labelled 1
       for the entire training set."""
    assert os.path.exists(train_label_path), "Training set not found."
    num1 = 0
    count = 0
    for file in os.listdir(train_label_path):
        count += 1
        labels = np.load(os.path.join(train_label_path, file))
        num1_ = np.count_nonzero(labels)
        num1 += num1_
    weight1 = (1 - num1 / (count * 27556)) ** 3.5    # 27556 = 166^2
    return np.array([1 - weight1, weight1])


def compute_iou(file):
    """Computes the IOU value of the predicted gate_1 according to the following equation:
       IOU = |{pred_in_gate}\cap {actual_in_gate}| / |{pred_in_gate}\cup {actual_in_gate}|
       The closer to 1, the better."""
    if file.endswith(".npy") or file.endswith(".csv"):
        raise ValueError("Do not include the file extension.")
    # load the files
    pred_gate = pd.read_csv(f"prediction__{file}.csv")["pred_gate_1"]
    actual_gate = pd.read_csv(f"../omiq_exported_data_processed/{file}.csv")["gate1_ir"]
    intersection, union = 0, 0
    for idx in range(pred_gate.size):
        if pred_gate[idx] == 1 or actual_gate[idx] == 1:
            union += 1
        if pred_gate[idx] == 1 and actual_gate[idx] == 1:
            intersection += 1
    print(f"{file} gate 1 prediction IOU = {intersection / union}")

