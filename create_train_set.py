from ctypes import sizeof
from lib2to3.pytree import convert
import os
import random
import pandas as pd
import numpy as np

from general_utils import convert_to_image

def main(args):
    val_size = args.val_size
    root = "D:\ShenLab\Cytometry\omiq_exported_data_processed"
    files = os.listdir(root)
    val_indices = random.sample(range(len(files)), val_size)
    
    for idx in range(len(files)):
        if idx % 10 == 0:
            print(f"converting {idx + 1}/{len(files)} files...")
        image = convert_to_image(root, files[idx], "train")
        label = convert_to_image(root, files[idx], "label")
        filename = files[idx][:-4]
        if idx in val_indices:
            np.save(f"./images/val_image/{filename}.npy", image)
            np.save(f"./images/val_label/{filename}.npy", label)
        else:
            np.save(f"./images/train_image/{filename}.npy", image)
            np.save(f"./images/train_label/{filename}.npy", label)
    print("Dataset creation completed.")


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch fcn training")

    parser.add_argument("-s", "--val-size", default=1, type=int, help="validation set size")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    if not os.path.exists("./images"):
        os.mkdir("./images")
        os.mkdir("./images/train_image")
        os.mkdir("./images/train_label")
        os.mkdir("./images/val_image")
        os.mkdir("./images/val_label")

    main(args)
