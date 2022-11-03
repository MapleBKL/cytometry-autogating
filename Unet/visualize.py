import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from scipy.spatial import ConvexHull

from general_utils import convert_to_image


def main(args):
    # The following directories should be set to fit your computer
    pred_root = "D:\\ShenLab\\Cytometry\\Unet\\prediction_results"
    label_root = "D:\\ShenLab\\Cytometry\\omiq_exported_data_processed"

    gate = args.gate
    if (not gate == 0) and (not gate == 1) and (not gate == 2):
        raise ValueError("Invalid gate number. If wish to visualise gate 1 (2), set gate to 1 (2). If wish to visualise both gates, do not set any value.")

    filename = args.filename
    if not filename:
        raise ValueError("Must designate a gate prediction to visualise.")
    if filename.endswith(".csv"):
        filename = filename[:-4]

    if args.filter_gate1:
        filter_gate1 = True
    else:
        filter_gate1 = False

    # If args.compare is set to True, show a side-by-side comparison of the
    # GT gate and the predicted gate.
    if args.compare:
        assert os.path.exists(label_root + f"\\{filename}.csv"), f"GT labels not found, cannot compare"
        # visualise only gate 1 or 2
        if gate == 1 or gate == 2:
            pred = convert_to_image(pred_root, f"prediction__{filename}.csv", gate, "visual", filter_gate1)
            actual = convert_to_image(label_root, f"{filename}.csv", gate, "visual", filter_gate1)

            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.suptitle(f"{filename}")
            ax1.imshow(actual)
            ax1.title.set_text(f"actual gate {gate}")
            ax2.imshow(pred)
            ax2.title.set_text(f"predicted gate {gate}")
            plt.show()
        # visualise both gates
        elif gate == 0:
            pred_1 = convert_to_image(pred_root, f"prediction__{filename}.csv", 1, "visual")
            actual_1 = convert_to_image(label_root, f"{filename}.csv", 1, "visual")
            pred_2 = convert_to_image(pred_root, f"prediction__{filename}.csv", 2, "visual", filter_gate1)
            actual_2 = convert_to_image(label_root, f"{filename}.csv", 2, "visual", filter_gate1)

            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
            fig.suptitle(f"{filename}")
            ax1.imshow(actual_1)
            ax1.title.set_text("actual gate 1")
            ax2.imshow(pred_1)
            ax2.title.set_text("predicted gate 1")
            ax3.imshow(actual_2)
            ax3.title.set_text("actual gate 2")
            ax4.imshow(pred_2)
            ax4.title.set_text("predicted gate 2")
            plt.show()

    # If args.compare is set to False, show a single high-resolution (not axes-converted)
    # image of the predicted gate.
    # In-gate data points are shown in blue, out-gate data points are shown in red,
    # and the convex hull of the gate is shown in black.
    else:
        pred = pd.read_csv(f"./prediction_results/prediction__{filename}.csv")

        if gate == 1 or gate == 2:
            if gate == 1:
                in_gate = pred[pred["gate1_ir"] == 1]
                out_gate = pred[pred["gate1_ir"] == 0]
                in_gate = in_gate[["Ir191Di___191Ir_DNA1", "Event_length"]].to_numpy()
                out_gate = out_gate[["Ir191Di___191Ir_DNA1", "Event_length"]].to_numpy()
                hull = ConvexHull(in_gate)
            
            elif gate == 2:
                in_gate = pred[pred["gate2_cd45"] == 1]
                out_gate = pred[pred["gate2_cd45"] == 0]
                in_gate = in_gate[["Ir193Di___193Ir_DNA2", "Y89Di___89Y_CD45"]].to_numpy()
                out_gate = out_gate[["Ir193Di___193Ir_DNA2", "Y89Di___89Y_CD45"]].to_numpy()
                hull = ConvexHull(in_gate)
                
            # make the plot
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.scatter(out_gate[:, 0], out_gate[:, 1], s=0.01, c='r')
            ax.scatter(in_gate[:, 0], in_gate[:, 1], s=0.01, c='b')
            for simplex in hull.simplices:
                plt.plot(in_gate[simplex, 0], in_gate[simplex, 1], 'k-')

            plt.show()
        
        elif gate == 0:
            in_gate_1 = pred[pred["gate1_ir"] == 1]
            out_gate_1 = pred[pred["gate1_ir"] == 0]
            in_gate_1 = in_gate_1[["Ir191Di___191Ir_DNA1", "Event_length"]].to_numpy()
            out_gate_1 = out_gate_1[["Ir191Di___191Ir_DNA1", "Event_length"]].to_numpy()
            hull_1 = ConvexHull(in_gate_1)

            in_gate_2 = pred[pred["gate2_cd45"] == 1]
            out_gate_2 = pred[pred["gate2_cd45"] == 0]
            in_gate_2 = in_gate_2[["Ir193Di___193Ir_DNA2", "Y89Di___89Y_CD45"]].to_numpy()
            out_gate_2 = out_gate_2[["Ir193Di___193Ir_DNA2", "Y89Di___89Y_CD45"]].to_numpy()
            hull_2 = ConvexHull(in_gate_2)

            fig, (ax1, ax2) = plt.subplots(1,2)
            fig.suptitle(f"{filename}")
            ax1.scatter(in_gate_1[:, 0], in_gate_1[:, 1], s=0.01, c='b')
            ax1.scatter(out_gate_1[:, 0], out_gate_1[:, 1], s=0.01, c='r')
            ax1.title.set_text("predicted gate 1")
            ax2.scatter(in_gate_2[:, 0], in_gate_2[:, 1], s=0.01, c='b')
            ax2.scatter(out_gate_2[:, 0], out_gate_2[:, 1], s=0.01, c='r')
            ax2.title.set_text("predicted gate 2")
            for simplex in hull_1.simplices:
                ax1.plot(in_gate_1[simplex, 0], in_gate_1[simplex, 1], 'k-')
            for simplex in hull_2.simplices:
                ax2.plot(in_gate_2[simplex, 0], in_gate_2[simplex, 1], 'k-')
            
            plt.show()
        
def parse_args():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("-g", "--gate", default=0, type=int)    # 0: both gates
    parser.add_argument("-f", "--filename", default=None, type=str)
    parser.add_argument("--compare", action=argparse.BooleanOptionalAction)
    parser.add_argument("--filter-gate1", action=argparse.BooleanOptionalAction, help="filter gate 1 when visualising gate 2")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    main(args)