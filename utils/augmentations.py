import numpy as np
from PIL import Image
import math
from tensorflow.keras.utils import Sequence
from albumentations import (Compose, OneOf,
                            CLAHE, Flip, Rotate, RGBShift, RandomContrast, RandomGamma, RandomBrightness,
                            Transpose, ShiftScaleRotate, RandomRotate90, OpticalDistortion, GridDistortion, ElasticTransform,
                            IAAPiecewiseAffine, ChannelShuffle, RandomBrightnessContrast, RandomGamma, RandomCrop)


def read_image(file_loc, dim=(256, 256)):
    img = Image.open(file_loc)
    img = img.resize(dim)
    img = np.array(img)
    return img


def read_mask(file_loc, dim=(256, 256)):
    img = Image.open(file_loc)
    img = img.resize(dim)
    img = np.array(img)
    img = (img > 0).astype(np.uint8)
    return img


class Train_Generator(Sequence):

    def __init__(self, x_set, y_set, batch_size=5, img_dim=(512, 512), augmentation=False):
        self.x = x_set
        self.y = y_set
        self.batch_size = batch_size
        self.img_dim = img_dim
        self.augmentation = augmentation

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    aug = Compose(
        [
            CLAHE(always_apply=True, p=1.0),

            OneOf([
                Flip(),
                Transpose()
            ], p=1.0),
            OneOf([
                ShiftScaleRotate(),
                RandomRotate90()
            ], p=0.9),

            OneOf([
                RGBShift(),
                RandomBrightnessContrast(),
                RandomGamma()
            ], p=0.2)
        ])

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_x = np.array([read_image(file_name, self.img_dim) for file_name in batch_x])
        batch_y = np.array([read_mask(file_name, self.img_dim) for file_name in batch_y])

        if self.augmentation is True:
            aug = [self.aug(image=i, mask=j) for i, j in zip(batch_x, batch_y)]
            batch_x = np.array([i['image'] for i in aug])
            batch_y = np.array([j['mask'] for j in aug])

        batch_y = np.expand_dims(batch_y, -1)

        return batch_x / 255.0, batch_y / 1.0


class Val_Generator(Sequence):

    def __init__(self, x_set, y_set, batch_size=5, img_dim=(512, 512), augmentation=False):
        self.x = x_set
        self.y = y_set
        self.batch_size = batch_size
        self.img_dim = img_dim
        self.augmentation = augmentation

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    aug = Compose(
        [
            CLAHE(always_apply=True, p=1.0)
        ])

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_x = np.array([read_image(file_name, self.img_dim) for file_name in batch_x])
        batch_y = np.array([read_mask(file_name, self.img_dim) for file_name in batch_y])

        if self.augmentation is True:
            aug = [self.aug(image=i, mask=j) for i, j in zip(batch_x, batch_y)]
            batch_x = np.array([i['image'] for i in aug])
            batch_y = np.array([j['mask'] for j in aug])
        batch_y = np.expand_dims(batch_y, -1)

        return batch_x / 255.0, batch_y / 1.0
