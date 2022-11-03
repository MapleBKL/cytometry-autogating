import os
import numpy as np
import pandas as pd
from general_utils import convert_to_image

root = "../omiq_exported_data_processed"
source = "./images/gate_1"
destination = "./images/gate_2"

for file in os.listdir(os.path.join(source, "train_image")):
    file = file[:-3]
    img = convert_to_image(root, file+"csv", 2, "train")
    np.save(os.path.join(destination + "/train_image", file+"npy"), img)
    lbl = convert_to_image(root, file+"csv", 2, "label")
    np.save(os.path.join(destination + "/train_label", file+"npy"), lbl)

for file in os.listdir(os.path.join(source, "val_image")):
    file = file[:-3]
    img = convert_to_image(root, file+"csv", 2, "train")
    np.save(os.path.join(destination + "/val_image", file+"npy"), img)
    lbl = convert_to_image(root, file+"csv", 2, "label")
    np.save(os.path.join(destination + "/val_label", file+"npy"), lbl)

for file in os.listdir(os.path.join(source, "test_image")):
    file = file[:-3]
    img = convert_to_image(root, file+"csv", 2, "train")
    np.save(os.path.join(destination + "/test_image", file+"npy"), img)
    lbl = convert_to_image(root, file+"csv", 2, "label")
    np.save(os.path.join(destination + "/test_label", file+"npy"), lbl)