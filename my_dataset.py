import os

import torch.utils.data as data
import torch
import numpy as np

from transforms import ToTensor


class GateSegmentation(data.Dataset):
    def __init__(self, transforms=None, mode="train"):
        super(GateSegmentation, self).__init__()
        # Change the following directory according to your computer
        root = "D:\ShenLab\Cytometry\preliminary_gating\plots"
        if mode == "train":
            self.image_dir = os.path.join(root, 'train')
            self.mask_dir = os.path.join(root, 'train_mask')
        elif mode == "val":
            self.image_dir = os.path.join(root, 'val')
            self.mask_dir = os.path.join(root, 'val_mask')

        self.images = []
        self.masks = []
        for filename in os.listdir(self.image_dir):
            self.images.append(filename)
        for filename in os.listdir(self.mask_dir):
            self.masks.append(filename)

        self.transforms = transforms

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = np.load(os.path.join(self.image_dir, self.images[index]))
        target_image = np.load(os.path.join(self.mask_dir, self.masks[index]))

        img = torch.as_tensor(img, dtype=torch.float)
        img = torch.permute(img, (2, 0, 1))
        target_image = torch.as_tensor(target_image, dtype=torch.int64)
        target_image = torch.permute(target_image, (2, 0, 1))
        target = torch.zeros((166, 166))
        for i in range(166):
            for j in range(166):
                target[i,j] = target_image[0,i,j]
        target = target.type(torch.LongTensor)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.images)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets


def cat_list(images, fill_value=0):
    # 计算该batch数据中，channel, h, w的最大值
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs


# dataset = GateSegmentation(voc_root="/data/", transforms=get_transform(train=True))
# d1 = dataset[0]
# print(d1)