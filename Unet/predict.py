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
    # The weights are best models by default
    weights_1_path = "./saved_weights/gate_1/" + args.weights_gate1
    weights_2_path = "./saved_weights/gate_2/" + args.weights_gate2
    assert os.path.exists(weights_1_path), f"weights {weights_1_path} not found."
    assert os.path.exists(weights_2_path), f"weights {weights_2_path} not found."
    filename = args.filename
    assert filename is not None, "Designate a file for autogating."
    if filename.endswith("csv"):
        filename = filename[:-4]

    file_path = "../omiq_exported_data_processed/" + filename + ".csv"
    assert os.path.exists(file_path), f"file {file_path} not found."

    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    model_1 = UNet(in_channels=1, num_classes=classes+1, base_c=32) # gate 1
    model_2 = UNet(in_channels=1, num_classes=classes+1, base_c=32) # gate 2

    # load weights
    model_1.load_state_dict(torch.load(weights_1_path, map_location='cpu')['model'])
    model_2.load_state_dict(torch.load(weights_2_path, map_location='cpu')['model'])
    model_1.to(device)
    model_2.to(device)

    file = pd.read_csv(file_path)

    # generate image input for gate 1
    file_1 = file[["Ir191Di___191Ir_DNA1", "Event_length"]].to_numpy()
    img_1 = np.zeros((166, 166, 1))
    for cell in file_1:
        ir191, el = cell
        # convert axes
        ir191 = int(ir191 * 10000 // 622)
        el = int(el - 10)
        img_1[ir191, el, 0] += 1
    # convert to tensor
    img_1 = torch.as_tensor(img_1, dtype=torch.float)

    # generate image input for gate 2
    file_2 = file[["Ir193Di___193Ir_DNA2", "Y89Di___89Y_CD45"]].to_numpy()
    img_2 = np.zeros((256, 256, 1))
    for cell in file_2:
        ir193, y89 = cell
        # convert axes
        ir193 = int(ir193 * 1000 // 43)
        y89 = int(y89 * 10000 // 330)
        img_2[ir193, y89, 0] += 1
    # convert to tensor
    img_2 = torch.as_tensor(img_2, dtype=torch.float)
    
    # convert the images to the format suitable for the network to process
    img_1 = torch.permute(img_1, (2,0,1))
    img_1 = torch.unsqueeze(img_1, dim=0)
    img_2 = torch.permute(img_2, (2,0,1))
    img_2 = torch.unsqueeze(img_2, dim=0)

    # predict gate 1 and gate 2
    pred_1 = autogate(1, model_1, img_1, device)
    pred_2 = autogate(2, model_2, img_2, device)

    # save the prediction results as a csv file
    print("Saving prediction results to csv file...")
    file_length = file.shape[0]
    gate_result = np.zeros((file_length, 6))

    for idx in range(file_length):
        ir191, el = file_1[idx]
        ir193, y89 = file_2[idx]
        # convert axes
        ir191_c = int(ir191 * 10000 // 622)
        el_c = int(el - 10)
        ir193_c =int(ir193 * 1000 // 43)
        y89_c = int(y89 * 10000 // 330)

        pred_gate_1 = pred_1[ir191_c, el_c].item()   # predicted gate 1 value
        pred_gate_2 = pred_2[ir193_c, y89_c].item()
        gate_result[idx] = ir191, el, ir193, y89, pred_gate_1, pred_gate_2 * pred_gate_1

    gate_result = pd.DataFrame(gate_result, columns=["Ir191Di___191Ir_DNA1", "Event_length", "Ir193Di___193Ir_DNA2", "Y89Di___89Y_CD45", "gate1_ir", "gate2_cd45"])
    gate_result.to_csv(f"./prediction_results/prediction__{filename}.csv", index=False)

    print(f"Prediction results saved to the location: ./prediction_results/prediction__{filename}.csv")

def autogate(gate, model, image, device):
    model.eval()
    with torch.no_grad():
        #initialise model
        if gate == 1:
            init_img = torch.zeros((1, 1, 166, 166), device=device)
        else:
            init_img = torch.zeros((1, 1, 256, 256), device=device)
        model(init_img)

        t_start = time_synchronized()
        output = model(image.to(device))
        t_end = time_synchronized()
        print(f"gate {gate} inference time: {t_end - t_start}")

        prediction = output['out'].argmax(1).squeeze(0)
        prediction = prediction.to("cpu").numpy().astype(np.uint8)
        prediction = np.expand_dims(prediction, 2)

        return prediction

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pregating with Unet")

    parser.add_argument("-f", "--filename", default=None, type=str, help="original csv filename")
    parser.add_argument("--weights-gate1", default="best_model.pth", type=str)
    parser.add_argument("--weights-gate2", default="best_model.pth", type=str)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists("./prediction_results"):
        os.mkdir("./prediction_results")

    main(args)
