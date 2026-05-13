import torch
from torchvision.transforms import v2 as transforms
from src.utils import misc

def train_transform() -> transforms.Compose:
    """
    Create the train transform
    :return: The transform
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.GaussianBlur(3),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=misc.IMAGE_NET_MEAN, std=misc.IMAGE_NET_STD)
    ])


def eval_transform() -> transforms.Compose:
    """
    Create the eval transform
    :return: The transform
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=misc.IMAGE_NET_MEAN, std=misc.IMAGE_NET_STD)
    ])

