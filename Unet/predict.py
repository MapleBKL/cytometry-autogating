import os
import time

import torch
import numpy as np
import pandas as pd

from src import UNet


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main(args):
    classes = 1
    # read in arguments and check validity
    weights_path = "./saved_weights/" + args.weights
    assert os.path.exists(weights_path), f"weights {weights_path} not found."
    if (args.image and args.filename) or (not args.image and not args.filename):
        raise ValueError("Provide either image or csv file.")
    filename = args.image or args.filename
    if args.image:
        img_path = "./images/val_image/" + args.image
        assert os.path.exists(img_path), f"image {img_path} not found."
    elif args.filename:
        file_path = "../omiq_exported_data_processed/" + args.filename
        assert os.path.exists(file_path), f"file {file_path} not found."

    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    model = UNet(in_channels=1, num_classes=classes+1, base_c=32)

    # load weights
    model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])
    model.to(device)

    # load image
    if args.image:
        img = torch.as_tensor(np.load(img_path), dtype=torch.float)

    # load csv file
    if args.filename:
        file = pd.read_csv(file_path)
        file = file[["Ir191Di___191Ir_DNA1", "Event_length"]].to_numpy()
        img = np.zeros((166, 166, 1))
        for cell in file:
            ir191, el = cell
            ir191 = int(ir191 * 10000 // 622)   # convert axes
            el = int(el - 10)   # convert axes
            img[ir191, el, 0] += 1
        img = torch.as_tensor(img, dtype=torch.float)
    
    # convert img to the format suitable for the network to process
    img = torch.permute(img, (2,0,1))
    img = torch.unsqueeze(img, dim=0)

    model.eval()    # turn on evaluation mode
    with torch.no_grad():
        # initialize model
        init_img = torch.zeros((1, 1, 166, 166), device=device)
        model(init_img)
      
        t_start = time_synchronized()
        output = model(img.to(device))
        t_end = time_synchronized()
        print("inference time: {}".format(t_end - t_start))

        prediction = output['out'].argmax(1).squeeze(0)
        prediction = prediction.to("cpu").numpy().astype(np.uint8)
        prediction = np.expand_dims(prediction, 2)
        np.save(f"prediction__{filename[:-4]}.npy", prediction)    # save a prediction image

        # if pass in a csv file, we also save the gating result
        if args.filename:
            file_length = file.shape[0]
            gate_result = np.zeros((file_length, 3))
            for idx in range(file_length):
                ir191, el = file[idx]
                ir191_c = int(ir191 * 10000 // 622)   # convert axes
                el_c = int(el - 10)   # convert axes
                pred_gate = prediction[ir191_c, el_c].item()   # predicted gating value
                gate_result[idx] = ir191, el, pred_gate
            gate_result = pd.DataFrame(gate_result, columns=["Ir191Di___191Ir_DNA1", "Event_length", "pred_gate_1"])
            gate_result.to_csv(f"prediction__{filename[:-4]}.csv")


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pregating with Unet")

    parser.add_argument("-i", "--image", default=None, type=str, help="image filename")
    parser.add_argument("-f", "--filename", default=None, type=str, help="original csv filename")
    parser.add_argument("-w", "--weights", default="best_model.pth", type=str)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
