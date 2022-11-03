import os
import time
import datetime
import random

import torch

from src import UNet
from train_utils import train_one_epoch, evaluate, create_lr_scheduler
from my_dataset import GateSegmentation


def create_model(num_classes):
    model = UNet(in_channels=1, num_classes=num_classes, base_c=32)
    return model

def verify(gate):
    """Verify the dataset for the given gate.
       Go through train_image and val_image, and check all the images have the
       corresponding labels in train_label and val_label, respectively."""
    train_imgs = os.listdir(f"./images/gate_{gate}/train_image")
    train_lbls = os.listdir(f"./images/gate_{gate}/train_label")
    assert len(train_imgs) == len(train_lbls), "Training images and training labels are not one-to-one."
    for filename in train_imgs:
        assert filename in train_lbls, f"Training image for {filename[:-4]} does not have a corresponding label."
    
    val_imgs = os.listdir(f"./images/gate_{gate}/val_image")
    val_lbls = os.listdir(f"./images/gate_{gate}/val_label")
    assert len(val_imgs) == len(val_lbls), "Validation images and training labels are not one-to-one."
    for filename in val_imgs:
        assert filename in val_lbls, f"Validation image for {filename[:-4]} does not have a corresponding label."

def main(args):
    gate = args.gate
    assert gate == 1 or gate == 2, print("Provide a gate: either 1 or 2")

    # verify all the training images and validation images have corresponding labels
    print("Verifying files...")
    if gate == 1:
        verify(1)
    else:
        verify(2)
    print("Verified.")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size
    # segmentation nun_classes + background
    num_classes = args.num_classes + 1
    
    # log information during training and validation
    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    train_dataset = GateSegmentation(gate=gate, mode="train")

    val_dataset = GateSegmentation(gate=gate, mode="val")

    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               num_workers=num_workers,
                                               shuffle=True,
                                               pin_memory=True,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             num_workers=num_workers,
                                             pin_memory=True,
                                             collate_fn=val_dataset.collate_fn)

    model = create_model(num_classes=num_classes)
    model.to(device)

    params_to_optimize = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.SGD(
        params_to_optimize,
        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # create learning rate update scheduler, it updates every step instead of every epoch
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    best_dice = 0.
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        mean_loss, lr = train_one_epoch(model, optimizer, train_loader, device, epoch, num_classes,
                                        lr_scheduler=lr_scheduler, print_freq=args.print_freq, scaler=scaler)

        confmat, dice = evaluate(model, val_loader, device=device, num_classes=num_classes)
        val_info = str(confmat)
        print(val_info)
        print(f"dice coefficient: {dice:.3f}")
        # write into txt
        with open(results_file, "a") as f:
            train_info = f"[epoch: {epoch}]\n" \
                         f"train_loss: {mean_loss:.4f}\n" \
                         f"lr: {lr:.6f}\n" \
                         f"dice coefficient: {dice:.3f}\n"
            f.write(train_info + val_info + "\n\n")

        if args.save_best:
            if best_dice < dice:
                best_dice = dice
            else:
                continue

        save_file = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "lr_scheduler": lr_scheduler.state_dict(),
                     "epoch": epoch,
                     "args": args}
        if args.amp:
            save_file["scaler"] = scaler.state_dict()

        if args.save_best:
            if gate == 1:
                torch.save(save_file, f"saved_weights/gate_1/best_model.pth")
            else:
                torch.save(save_file, f"saved_weights/gate_2/best_model.pth")
        else:
            if gate == 1:
                torch.save(save_file, "saved_weights/gate_1/model_{}.pth".format(epoch))
            else:
                torch.save(save_file, "saved_weights/gate_2/model_{}.pth".format(epoch))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="unet cytometry autogating")

    parser.add_argument("-g", "--gate", default=0, type=int)
    parser.add_argument("--num-classes", default=1, type=int)   # DO NOT CHANGE!
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("-b", "--batch-size", default=4, type=int)
    parser.add_argument("--epochs", default=50, type=int, metavar="N",
                        help="number of total epochs to train")
    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--save-best', action=argparse.BooleanOptionalAction, help='only save best dice weights')
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction,
                        help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists("./saved_weights"):
        os.mkdir("./saved_weights")
        os.mkdir("./saved_weights/gate_1")
        os.mkdir("./saved_weights/gate_2")

    main(args)