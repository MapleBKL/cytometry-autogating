import os
import random
import pandas as pd
import numpy as np

from general_utils import convert_to_image

def main(args):
    val_size = args.val_size
    root = "../../omiq_exported_data_processed"
    files = os.listdir(root)
    val_indices = random.sample(range(len(files)), val_size)
    print(f"The following {val_size} files will be the validation set:")
    for i in range(val_size):
        print(files[val_indices[i]])
    print("--------------------------------------------------")
    
    # Convert all the files in the root directory.
    for idx in range(len(files)):
        if idx % 10 == 0:
            print(f"converting {idx + 1}/{len(files)} files...")
        image = convert_to_image(root, files[idx], mode="train")
        label = convert_to_image(root, files[idx], mode="label")
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
    parser = argparse.ArgumentParser(description="create train/validation sets")

    parser.add_argument("-s", "--val-size", default=1, type=int, help="validation set size")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    # The train/validation sets will be stored in the folders created below.
    os.mkdir("./images")
    os.mkdir("./images/train_image")
    os.mkdir("./images/train_label")
    os.mkdir("./images/val_image")
    os.mkdir("./images/val_label")

    main(args)
