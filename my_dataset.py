import os

import torch.utils.data as data
import torch
import numpy as np


class GateSegmentation(data.Dataset):
    def __init__(self, mode="train"):
        super(GateSegmentation, self).__init__()
        # Change the following directory according to your computer
        root = "D:\ShenLab\Cytometry\preliminary_gating\plots"
        if mode == "train":
            self.image_dir = os.path.join(root, 'train')
            self.label_dir = os.path.join(root, 'train_label')
        elif mode == "val":
            self.image_dir = os.path.join(root, 'val')
            self.label_dir = os.path.join(root, 'val_label')

        self.images = []
        self.labels = []
        for filename in os.listdir(self.image_dir):
            self.images.append(filename)
        for filename in os.listdir(self.label_dir):
            self.labels.append(filename)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = np.load(os.path.join(self.image_dir, self.images[index]))
        target = np.load(os.path.join(self.label_dir, self.labels[index]))

        img = torch.as_tensor(img, dtype=torch.float)
        img = torch.permute(img, (2, 0, 1))
        target = torch.as_tensor(target, dtype=torch.int64)

        return img, target

    def __len__(self):
        return len(self.images)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets

# this function has no effect in our project
def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs
