import matplotlib.pyplot as plt
import numpy as np

def main(args):
    filename = args.file
    pred = np.load("prediction.npy")
    actual = np.expand_dims(np.load(f"./images/train_label/{filename}"), 2)

    _, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(actual)
    ax2.imshow(pred)
    plt.show()

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("-f", "--file", default=None, type=str)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    main(args)