import random
import torch.utils.data
import torchvision.datasets as dset

from torchvision import transforms
from PIL import Image
import os
import torchvision.transforms.functional as F

from .logger import Logger

IMAGE_SIZE = 64
DSET_MEAN = torch.tensor([0.8516, 0.8513, 0.8511])
DSET_STD = torch.tensor([0.2303, 0.2305, 0.2310])
logger = Logger(__name__)


def get_transforms():
    return [
        transforms.Lambda(white_square_padding),
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            0.05, 0.05, 0.1, 0.5),
        transforms.Lambda(random_adjust_sharpness),
        transforms.ToTensor()
    ]

def white_square_padding(img):
    w, h = img.size

    if w < h:
        margin = (h - w) // 2
        padding = (margin, 0, margin, 0)
    else:
        margin = (w - h) // 2
        padding = (0, margin, 0, margin)

    return F.pad(img, padding, 255, 'constant')


def minus_one_to_one(tensor):
    tmin = tensor.min()
    tmax = tensor.max()

    return ((tensor - tmin) / (tmax - tmin)) * 2 - 1


def random_adjust_sharpness(img):
    if random.random() < 0.5:
        return F.adjust_sharpness(img, random.uniform(0.8, 1.2))
    else:
        return img


def dataloader(dataroot, batch_size=128, dataloader_workers=12, recalculate_mean_std=False):
    if recalculate_mean_std:
        mean, std = calculate_mean_std(dataroot, dataloader_workers)
    else:
        mean, std = DSET_MEAN, DSET_STD

    transforms_with_normalization = get_transforms()
    transforms_with_normalization.append(transforms.Normalize(mean, std))
    transforms_with_normalization.append(transforms.Lambda(torch.tanh))

    dataset = dset.ImageFolder(
        root=dataroot, transform=transforms.Compose(transforms_with_normalization))

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=dataloader_workers, pin_memory=True)

    return dataloader


def replace_transparent_images(directory):
    n_converted = 0

    for filename in os.listdir(directory):
        full_path = directory / filename

        with Image.open(full_path) as im:
            if im.mode in ('RGBA', 'LA') or (im.mode == 'P' and 'transparency' in im.info):
                logger.info(
                    f"[{n_converted}] Removing transparency from {filename}")

                alpha = im.convert('RGBA').getchannel('A')

                bg = Image.new("RGBA", im.size, (255, 255, 255))
                bg.paste(im, mask=alpha)
                bg.convert('RGB').save(
                    directory / (full_path.stem + ".jpg"), "JPEG", quality=80)
                bg.close()

                os.remove(full_path)
                n_converted += 1


def calculate_mean_std(dataroot, dataloader_workers=12):
    dataset = dset.ImageFolder(
        root=dataroot, transform=transforms.Compose(get_transforms()))

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=128, shuffle=False, num_workers=dataloader_workers)

    mean = 0.
    std = 0.

    for images, _ in loader:
        # batch size (the last batch can have smaller size!)
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)

    mean /= len(loader.dataset)
    std /= len(loader.dataset)

    # Mean: tensor([0.8414, 0.8411, 0.8409]), Std: tensor([0.2244, 0.2246, 0.2252])
    logger.info(
        f"Calculated mean and std of the data in {dataroot}: Mean: {mean}, Std: {std}")

    return mean, std
