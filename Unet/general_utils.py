import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import inf
import os


def global_extrema(dir_name, gate=0):
    """Return the global maxima and minima of the relevant parameters for the designated gate
       in all the files under the given directory.
       dir_name: the directory holding all the csv files
       gate: 1 or 2"""
    # The following helper function returns the relavant maxima and minima
    # for a specific file
    assert gate == 1 or gate == 2, "Provide a gate: either 1 or 2."
    def local_extrema(filename, gate):
        data = pd.read_csv(os.path.join(dir_name, filename))
        if gate == 1:
            ir191 = data["Ir191Di___191Ir_DNA1"]
            el = data["Event_length"]
            return ir191.max(), ir191.min(), el.max(), el.min()
        elif gate == 2:
            ir193 = data["Ir193Di___193Ir_DNA2"]
            y89 = data["Y89Di___89Y_CD45"]
            return ir193.max(), ir193.min(), y89.max(), y89.min()
    x_max, y_max = -inf, -inf
    x_min, y_min = inf, inf
    # find the global maxima and minima for the two parameters across
    # all the files in the directory
    for file in os.listdir(dir_name):
        x_max_l, x_min_l, y_max_l, y_min_l = local_extrema(file, gate)
        x_max = max(x_max, x_max_l)
        x_min = min(x_min, x_min_l)
        y_max = max(y_max, y_max_l)
        y_min = min(y_min, y_min_l)
    return x_max, x_min, y_max, y_min


# NOTE: the following conversion depends on the result of the function 'global_extrema' 
def axes_convert(df_numpy, gate=0):
    """Convert the axes according to the following rules:
       gate 1:
       ir191: ir191 * 10000 // 622
       el: el - 10
       gate 2:
       ir193: ir193 * 1000 // 43
       y89: y89 * 10000 // 330
       The input is a numpy array of shape (cell_numbers, 3). The output is a numpy
       array of the same shape as the input."""
    if gate == 1:
        for idx in range(df_numpy.shape[0]):
            ir191, el, gate1 = df_numpy[idx]
            ir191 = ir191 * 10000 // 622
            el = el - 10
            df_numpy[idx] = np.array((ir191, el, gate1))
        return df_numpy
    elif gate == 2:
        for idx in range(df_numpy.shape[0]):
            ir193, y89, gate2 = df_numpy[idx]
            ir193 = ir193 * 1000 // 43
            y89 = y89 * 10000 // 330
            df_numpy[idx] = np.array((ir193, y89, gate2))
        return df_numpy


def convert_to_image(dir_name, filename, gate=0, mode="visual", filter_gate1=False):
    """Convert the original csv files to images for visualising the gate (1 or 2), training, or GT labelling.
       modes:
       visual (default) - to visualise the gate
       train - to produce an image for training / predicting
       label - to produce a mask image
       
       filter_gate1: if set to False, visual mode for gate 2 shows all the cells; if set to True, visual mode
       for gate 2 shows only the cells in gate 1. Default is False.
       """
    assert gate == 1 or gate == 2, "Provide a gate: either 1 or 2."

    if mode == "visual" and gate == 2 and not filter_gate1:
        data = pd.read_csv(os.path.join(dir_name, filename))[["Ir193Di___193Ir_DNA2", "Y89Di___89Y_CD45", "gate2_cd45"]]
        data = axes_convert(data.to_numpy(), gate)
        image = np.zeros((256, 256, 1))
        return generate_visual(data, image)

    if gate == 1:
        data = pd.read_csv(os.path.join(dir_name, filename))[["Ir191Di___191Ir_DNA1", "Event_length", "gate1_ir"]]
    elif gate == 2:
        data = pd.read_csv(os.path.join(dir_name, filename))[["Ir193Di___193Ir_DNA2", "Y89Di___89Y_CD45", "gate1_ir", "gate2_cd45"]]
        # filter out gate 1
        data = data[data["gate1_ir"] == 1]
        data = data[["Ir193Di___193Ir_DNA2", "Y89Di___89Y_CD45", "gate2_cd45"]]
    data = axes_convert(data.to_numpy(), gate)

    if mode == "visual" or mode == "train":
        if gate == 1:
            image = np.zeros((166, 166, 1))
        else:
            image = np.zeros((256, 256, 1))
        if mode == "visual":
            return generate_visual(data, image)
        elif mode == "train":
            return generate_train(data, image)

    elif mode == "label":
        if gate == 1:
            label = np.zeros((166, 166))
        else:
            label = np.zeros((256, 256))
        return generate_label(data, label)
    

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
    """Computes the IOU values of the predicted gates according to the following equation:
       IOU = |{pred_in_gate}\cap {actual_in_gate}| / |{pred_in_gate}\cup {actual_in_gate}|
       The closer to 1, the better."""
    if file.endswith(".csv"):
        file = file[:-4]
    
    pred_gate_1 = pd.read_csv(f"./prediction_results/prediction__{file}.csv")["gate1_ir"]
    actual_gate_1 = pd.read_csv(f"../omiq_exported_data_processed/{file}.csv")["gate1_ir"]
    intersection, union = 0, 0
    for idx in range(pred_gate_1.size):
        if pred_gate_1[idx] == 1 or actual_gate_1[idx] == 1:
            union += 1
        if pred_gate_1[idx] == 1 and actual_gate_1[idx] == 1:
            intersection += 1
    iou_1 = intersection / union

    pred_gate_2 = pd.read_csv(f"./prediction_results/prediction__{file}.csv")["gate2_cd45"]
    actual_gate_2 = pd.read_csv(f"../omiq_exported_data_processed/{file}.csv")["gate2_cd45"]
    intersection, union = 0, 0
    for idx in range(pred_gate_2.size):
        if pred_gate_2[idx] == 1 or actual_gate_2[idx] == 1:
            union += 1
        if pred_gate_2[idx] == 1 and actual_gate_2[idx] == 1:
            intersection += 1
    iou_2 = intersection / union

    print(f"{file} gate 1 prediction IOU = {iou_1}")
    print(f"{file} gate 2 prediction IOU = {iou_2}")

def compute_dice(file):
    """Computes the dice coefficients of the predicted gates according to the following equation:
       dice = 2*|{pred_in_gate}\cap {actual_in_gate}|/(|{pred_in_gate}|+|{actual_in_gate}|)
       The closer to 1, the better."""
    if file.endswith(".csv"):
        file = file[:-4]

    pred_gate_1 = pd.read_csv(f"./prediction_results/prediction__{file}.csv")["gate1_ir"]
    actual_gate_1 = pd.read_csv(f"../omiq_exported_data_processed/{file}.csv")["gate1_ir"]
    intersection, pred_in_gate, actual_in_gate = 0, 0, 0
    for idx in range(pred_gate_1.size):
        if pred_gate_1[idx] == 1:
            pred_in_gate += 1
        if actual_gate_1 == 1:
            actual_in_gate += 1
        if pred_gate_1[idx] == 1 and actual_gate_1[idx] == 1:
            intersection += 1
    dice_1 = 2 * intersection / (pred_in_gate + actual_in_gate)

    pred_gate_2 = pd.read_csv(f"./prediction_results/prediction__{file}.csv")["gate2_cd45"]
    actual_gate_2 = pd.read_csv(f"../omiq_exported_data_processed/{file}.csv")["gate2_cd45"]
    intersection, pred_in_gate, actual_in_gate = 0, 0, 0
    for idx in range(pred_gate_2.size):
        if pred_gate_2[idx] == 1:
            pred_in_gate += 1
        if actual_gate_2 == 1:
            actual_in_gate += 1
        if pred_gate_2[idx] == 1 and actual_gate_2[idx] == 1:
            intersection += 1
    dice_2 = 2 * intersection / (pred_in_gate + actual_in_gate)

    print(f"{file} gate 1 dice coefficient = {dice_1}")
    print(f"{file} gate 2 dice coefficient = {dice_2}")


def plot_diff(file, gate):
    if gate == 1:
        pred_gate = pd.read_csv(f"./prediction_results/prediction__{file}.csv")[["Ir191Di___191Ir_DNA1", "Event_length", "gate1_ir"]]
        actual_gate = pd.read_csv(f"../omiq_exported_data_processed/{file}.csv")[["Ir191Di___191Ir_DNA1", "Event_length", "gate1_ir"]]
        pred_gate = axes_convert(pred_gate.to_numpy(), 1)
        actual_gate = axes_convert(actual_gate.to_numpy(), 1)
        diff = np.zeros((166, 166, 1))
        for idx in range(len(pred_gate)):
            x, y, p_gate = pred_gate[idx]
            _, _, a_gate = actual_gate[idx]
            x, y, p_gate, a_gate = int(x), int(y), int(p_gate), int(a_gate)
            diff[x, y, 0] = p_gate - a_gate
        plt.imshow(diff)
        plt.show()
    else:
        pred_gate = pd.read_csv(f"./prediction_results/prediction__{file}.csv")[["Ir193Di___193Ir_DNA2", "Y89Di___89Y_CD45", "gate2_cd45"]]
        actual_gate = pd.read_csv(f"../omiq_exported_data_processed/{file}.csv")[["Ir193Di___193Ir_DNA2", "Y89Di___89Y_CD45", "gate2_cd45"]]
        pred_gate = axes_convert(pred_gate.to_numpy(), 2)
        actual_gate = axes_convert(actual_gate.to_numpy(), 2)
        diff = np.zeros((256, 256, 1))
        for idx in range(len(pred_gate)):
            x, y, p_gate = pred_gate[idx]
            _, _, a_gate = actual_gate[idx]
            x, y, p_gate, a_gate = int(x), int(y), int(p_gate), int(a_gate)
            diff[x, y, 0] = p_gate - a_gate
        plt.imshow(diff)
        plt.show()


# helper functions for convert_to_image:
def generate_visual(data, image):
    for cell in data:
        x, y, gate = cell
        x, y, gate = int(x), int(y), int(gate)
        if gate == 0:
            image[x, y, 0] = 128
        elif gate == 1:
            image[x, y, 0] = 255
    return image

def generate_train(data, image):
    for cell in data:
        x, y, _ = cell
        x, y = int(x), int(y)
        image[x, y, 0] += 1
    return image

def generate_label(data, label):
    for cell in data:
        x, y, gate = cell
        x, y, gate = int(x), int(y), int(gate)
        if gate == 1:
            label[x, y] = 1
    return label
