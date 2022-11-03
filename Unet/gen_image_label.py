import os
import numpy as np

from general_utils import convert_to_image

def main(args):
    gate = args.gate
    if (not gate == 0) and (not gate == 1) and (not gate == 2):
        raise ValueError("Invalid gate number. If wish to generate images and labels for gate 1 (2), set gate to 1 (2). To generate for both gates, do not set any value.")
    root = "../omiq_exported_data_processed"
    files = os.listdir(root)
    if gate == 1:
        convert(1, root, files)
    elif gate == 2:
        convert(2, root, files)
    else:
        convert(1, root, files)
        convert(2, root, files)

def convert(gate, root, files):
    if not os.path.exists(f"./all_images/gate_{gate}"):
        os.mkdir(f"./all_images/gate_{gate}")
        os.mkdir(f"./all_images/gate_{gate}/image")
        os.mkdir(f"./all_images/gate_{gate}/label")
    print(f"Generating images and labels for gate {gate}.")
    for idx in range(len(files)):
        if idx % 10 == 0:
            print(f"converting {idx + 1}/{len(files)} files...")
        image = convert_to_image(root, files[idx], gate=gate, mode="train")
        label = convert_to_image(root, files[idx], gate=gate, mode="label")
        filename = files[idx][:-4]
        np.save(f"./all_images/gate_{gate}/image/{filename}.npy", image)
        np.save(f"./all_images/gate_{gate}/label/{filename}.npy", label)
    print("Generation completed.")
        

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="create density images and labels")

    parser.add_argument("-g", "--gate", default=0, type=int, help="which gate to generate")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    if not os.path.exists("./all_images"):
        os.mkdir("./all_images")
    
    main(args)