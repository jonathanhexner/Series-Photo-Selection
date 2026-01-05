from torch.utils import data
from torchvision import transforms
import numpy as np
import torch
from PIL import Image
from data import make_shuffle_path


class Normalize(object):
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, img):

        img = np.array(img).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std

        return img


class ToTensor(object):
    def __call__(self, img):
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        img = torch.from_numpy(img).float()

        return img


class FixScaleCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, img):
        w, h = img.size
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        # center crop
        w, h = img.size
        x1 = int(round((w - self.crop_size) / 2.))
        y1 = int(round((h - self.crop_size) / 2.))
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return img


def transform(sample):
    composed_transforms = transforms.Compose([
        FixScaleCrop(crop_size=224),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensor()])
    return composed_transforms(sample)


class MyDataset(data.Dataset):

    def __init__(self, train=True, image_root=r'D:\Similar Images\automatic_triage_photo_series\train_val\train_val_imgs', seed=None):
        if train:
            self.pathA, self.pathB, self.result, _, _, _ = make_shuffle_path.make_shuffle_path(seed=seed)
        else:
            _, _, _, self.pathA, self.pathB, self.result = make_shuffle_path.make_shuffle_path(seed=seed)
        self.train = train
        self.image_root = image_root

    def __getitem__(self, index):
        import os
        imageA_path = os.path.join(self.image_root, self.pathA[index])
        imageB_path = os.path.join(self.image_root, self.pathB[index])

        imageA = Image.open(imageA_path).convert('RGB')
        imageB = Image.open(imageB_path).convert('RGB')

        imageA = transform(imageA)
        imageB = transform(imageB)
        return {
            'img1': imageA,
            'img2': imageB,
            'winner': int(self.result[index]),
            'image1': self.pathA[index],
            'image2': self.pathB[index]
        }

    def __len__(self):
        return len(self.result)


def make_loader(batch_size=8, num_workers=0, seed=None):
    """
    Create data loaders for training and validation.

    Args:
        batch_size: Batch size for training (default 8 for CPU)
        num_workers: Number of data loading workers (default 0 for Windows CPU)
        seed: Optional random seed for reproducible data shuffling (default: None)
    """
    train_data = MyDataset(train=True, seed=seed)
    val_data = MyDataset(train=False, seed=seed)
    # Set pin_memory=False for CPU, num_workers=0 to avoid Windows multiprocessing issues
    trainloader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True,
                                 num_workers=num_workers, pin_memory=False)
    valloader = data.DataLoader(val_data, batch_size=batch_size, shuffle=False,
                               num_workers=num_workers, pin_memory=False)
    return train_data, val_data, trainloader, valloader