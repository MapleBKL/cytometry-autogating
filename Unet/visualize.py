import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.spatial import ConvexHull

from general_utils import convert_to_image


def main(args):
    filename = args.filename
    if not filename:
        raise ValueError("Must designate a gate prediction to visualise.")
    if filename.endswith(".npy") or filename.endswith(".csv"):
        raise ValueError("Do not include the file extension.")

    # If args.compare is set to True, show a side-by-side comparison of the
    # GT gate and the predicted gate.
    if args.compare:  
    # The following directories should be set to fit your computer
        pred_img = convert_to_image("D:\\_Files\\Shen_Lab\\Cytometry\\Unet\\prediction_results\\labels", f"prediction__{filename}.csv", "pred_gate_1")
        actual_img = convert_to_image("D:\\_Files\\Shen_Lab\\Cytometry\\omiq_exported_data_processed", f"{filename}.csv")

        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle(f"{filename}")
        ax1.imshow(actual_img)
        ax1.title.set_text("actual gate 1")
        ax2.imshow(pred_img)
        ax2.title.set_text("predicted gate 1")
        plt.show()

    # If args.compare is set to False, show a single high-resolution (not axes-converted)
    # image of the predicted gate 1.
    # In-gate data points are shown in blue, out-gate data points are shown in red,
    # and the convex hull of the gate is shown in black.
    elif not args.compare:
        pred = pd.read_csv(f"./prediction_results/labels/prediction__{filename}.csv")
        in_gate = pred[pred["pred_gate_1"] == 1]
        out_gate = pred[pred["pred_gate_1"] == 0]
        in_gate = in_gate[["Ir191Di___191Ir_DNA1", "Event_length"]].to_numpy()
        out_gate = out_gate[["Ir191Di___191Ir_DNA1", "Event_length"]].to_numpy()
        hull = ConvexHull(in_gate)
        
        # make the plot
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(in_gate[:, 0], in_gate[:, 1], s=0.01, c='b')
        ax.scatter(out_gate[:, 0], out_gate[:, 1], s=0.01, c='r')
        for simplex in hull.simplices:
            plt.plot(in_gate[simplex, 0], in_gate[simplex, 1], 'k-')

        plt.show()
        
def parse_args():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("-f", "--filename", default=None, type=str)
    parser.add_argument("-c", "--compare", default=False, type=bool, help="compare with GT label")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    main(args)