import os
import torch

from src import fcn_resnet50
from train_utils import evaluate
from my_dataset import GateSegmentation


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    assert os.path.exists(args.weights), f"weights {args.weights} not found."

    # segmentation nun_classes + background
    num_classes = args.num_classes + 1

    val_dataset = GateSegmentation(transforms=None, mode="val")

    num_workers = 0
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             num_workers=num_workers,
                                             pin_memory=True,
                                             collate_fn=val_dataset.collate_fn)

    model = fcn_resnet50(aux=args.aux, num_classes=num_classes)
    model.load_state_dict(torch.load(args.weights, map_location=device)['model'])
    model.to(device)

    confmat = evaluate(model, val_loader, device=device, num_classes=num_classes)
    print(confmat)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch fcn training")

    parser.add_argument("--weights", default="./save_weights/model_0.pth")
    parser.add_argument("--num-classes", default=1, type=int)
    parser.add_argument("--aux", default=False, type=bool, help="auxilier loss")
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists("./saved_weights"):
        os.mkdir("./saved_weights")

    main(args)