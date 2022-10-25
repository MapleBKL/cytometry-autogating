import matplotlib.pyplot as plt
import numpy as np

from general_utils import convert_to_image


def main(args):
    filename = args.filename
    if filename.endswith(".npy") or filename.endswith(".csv"):
        raise ValueError("Do not include the file extension.")

    # The following directories should be set to fit your computer
    pred_img = convert_to_image("D:\\ShenLab\\Cytometry\\Unet", f"prediction__{filename}.csv", "pred_gate_1")
    actual_img = convert_to_image("D:\\ShenLab\\Cytometry\\omiq_exported_data_processed", filename+".csv")

    _, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(actual_img)
    ax1.title.set_text("actual gate 1")
    ax2.imshow(pred_img)
    ax2.title.set_text("predicted gate 1")
    plt.show()

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("-f", "--filename", default=None, type=str)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    main(args)