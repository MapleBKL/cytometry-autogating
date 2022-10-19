import os
import time
import json

import torch
import numpy as np

from src import fcn_resnet50


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main(args):
    assert args.image is not None, "Must provide image filename."
    aux = False  # inference time not need aux_classifier
    classes = 1
    root = "./images/val_image/"
    weights_path = "./saved_weights/" + args.weights
    img_path = root + args.image
    palette_path = "./palette.json"
    assert os.path.exists(weights_path), f"Weights {weights_path} not found."
    assert os.path.exists(img_path), f"Image {img_path} not found."
    assert os.path.exists(palette_path), f"Palette {palette_path} not found."
    with open(palette_path, "rb") as f:
        pallette_dict = json.load(f)
        pallette = []
        for v in pallette_dict.values():
            pallette += v

    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    model = fcn_resnet50(aux=aux, num_classes=classes+1)

    # delete weights about aux_classifier
    weights_dict = torch.load(weights_path, map_location='cpu')['model']
    for k in list(weights_dict.keys()):
        if "aux" in k:
            del weights_dict[k]

    # load weights
    model.load_state_dict(weights_dict)
    model.to(device)

    # load image
    img = torch.as_tensor(np.load(img_path), dtype=torch.float)
    img = torch.permute(img, (2, 0, 1))
    img = torch.unsqueeze(img, dim=0)

    model.eval()  # enter evaluation mode
    with torch.no_grad():
        # init model
        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, 1, img_height, img_width), device=device)
        model(init_img)

        t_start = time_synchronized()
        output = model(img.to(device))
        t_end = time_synchronized()
        print("inference time: {}".format(t_end - t_start))

        prediction = output['out'].argmax(1).squeeze(0)
        prediction = prediction.to("cpu").numpy().astype(np.uint8)
        prediction = np.expand_dims(prediction, 2)
        np.save("prediction.npy", prediction)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch fcn training")

    parser.add_argument("-i", "--image", default=None, type=str, help="image filename")
    parser.add_argument("-w", "--weights", default="model_0.pth", type=str)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    main(args)