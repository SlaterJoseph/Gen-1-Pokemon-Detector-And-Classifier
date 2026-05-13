from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from src.utils import paths
from sklearn.model_selection import train_test_split

def build_loader(dataset: ImageFolder, batch_size: int=128, shuffle: bool=True, num_workers: int=4) -> DataLoader:
    """
    Build the dataloader
    :param dataset: The given dataset
    :param batch_size: The batch size
    :param shuffle: If the dataset should be shuffled
    :param num_workers: The amount of workers to use
    :return: The dataloader of the dataset
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return loader


def split_dataset(dataset: ImageFolder, test_size: float=0.1, seed: int=42) -> tuple[list[int], list[int], list[int]]:
    """
    Splits the dataset into train, validation, and test indices.
    :param dataset: The dataset
    :param test_size: The size that the test set will be
    :param seed: Seed for reproducibility
    :return: A tuple of train, validation, and test indices
    """
    indices = list(range(len(dataset)))
    labels = dataset.targets

    train_val, test_idx = train_test_split(indices, test_size=test_size, stratify=labels, random_state=seed)

    train_val_labels = [labels[i] for i in train_val]
    train_idx, val_idx = train_test_split(train_val, test_size=(test_size * 1.1), stratify=train_val_labels, random_state=seed)

    return train_idx, val_idx, test_idx


def get_subsets(train_transformer, eval_transformer, test_size: float=0.1, seed: int=42) -> tuple[Subset, Subset, Subset]:
    """
    Returns the subsets of each category of data
    :param train_transformer: The transformer to use for the training set
    :param eval_transformer: The transformer to use for the test set
    :param test_size: The size of the test set
    :param seed: The seed for reproducibility
    :return: A tuple of the train, test, and validation sets
    """

    train_dataset = ImageFolder(paths.TRAINING_DATA, transform=train_transformer)
    eval_dataset = ImageFolder(paths.TRAINING_DATA, transform=eval_transformer)

    train_idx, val_idx, test_idx = split_dataset(train_dataset, test_size, seed=seed)

    train_subset = Subset(train_dataset, train_idx)
    val_subset = Subset(eval_dataset, val_idx)
    test_subset = Subset(eval_dataset, test_idx)

    return train_subset, val_subset, test_subset
