from __future__ import annotations
from collections import Counter

import pytest
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

from src.classifier.dataset import build_loader, split_dataset, get_subsets



def test_split_dataset_returns_three_index_lists(tiny_image_folder):
    ds = ImageFolder(str(tiny_image_folder))
    train_idx, val_idx, test_idx = split_dataset(ds, test_size=0.1, seed=42)

    assert isinstance(train_idx, list)
    assert isinstance(val_idx, list)
    assert isinstance(test_idx, list)
    assert all(isinstance(i, int) for i in train_idx[:5])


def test_split_dataset_covers_all_samples_exactly_once(tiny_image_folder):
    ds = ImageFolder(str(tiny_image_folder))
    train_idx, val_idx, test_idx = split_dataset(ds, test_size=0.1, seed=42)

    combined = sorted(train_idx + val_idx + test_idx)
    assert combined == list(range(len(ds))), "splits should be a partition of all indices"


def test_split_dataset_no_overlap(tiny_image_folder):
    ds = ImageFolder(str(tiny_image_folder))
    train_idx, val_idx, test_idx = split_dataset(ds, test_size=0.1, seed=42)

    assert not (set(train_idx) & set(val_idx))
    assert not (set(train_idx) & set(test_idx))
    assert not (set(val_idx) & set(test_idx))


def test_split_dataset_reproducible(tiny_image_folder):
    ds = ImageFolder(str(tiny_image_folder))
    a = split_dataset(ds, test_size=0.1, seed=42)
    b = split_dataset(ds, test_size=0.1, seed=42)
    assert a == b, "same seed should produce the same split"


def test_split_dataset_different_seeds_diverge(tiny_image_folder):
    ds = ImageFolder(str(tiny_image_folder))
    a = split_dataset(ds, test_size=0.1, seed=1)
    b = split_dataset(ds, test_size=0.1, seed=2)
    assert a != b


def test_split_dataset_stratified_each_class_in_each_split(tiny_image_folder):
    ds = ImageFolder(str(tiny_image_folder))
    train_idx, val_idx, test_idx = split_dataset(ds, test_size=0.125, seed=42)

    labels = ds.targets
    classes = set(labels)

    train_classes = {labels[i] for i in train_idx}
    val_classes = {labels[i] for i in val_idx}
    test_classes = {labels[i] for i in test_idx}

    assert train_classes == classes, "missing classes in train split"
    assert val_classes == classes, "missing classes in val split"
    assert test_classes == classes, "missing classes in test split"



def test_get_subsets_returns_three_subsets(tiny_image_folder, monkeypatch):
    from src.utils import paths
    monkeypatch.setattr(paths, "TRAINING_DATA", str(tiny_image_folder))

    train_s, val_s, test_s = get_subsets(
        train_transformer=None, eval_transformer=None, test_size=0.125, seed=42,
    )

    assert isinstance(train_s, Subset)
    assert isinstance(val_s, Subset)
    assert isinstance(test_s, Subset)


def test_get_subsets_uses_eval_transform_for_val_and_test(tiny_image_folder, monkeypatch):
    from src.utils import paths
    monkeypatch.setattr(paths, "TRAINING_DATA", str(tiny_image_folder))

    train_transformer = "TRAIN_SENTINEL"
    eval_transformer = "EVAL_SENTINEL"

    train_s, val_s, test_s = get_subsets(
        train_transformer=train_transformer,
        eval_transformer=eval_transformer,
        seed=42,
    )
    assert val_s.dataset is test_s.dataset, "val and test must share an underlying dataset"
    assert train_s.dataset is not val_s.dataset, "train must use a different dataset (different transform)"


def test_build_loader_returns_dataloader(tiny_image_folder):
    ds = ImageFolder(str(tiny_image_folder))
    loader = build_loader(ds, batch_size=4, shuffle=False, num_workers=0)

    assert isinstance(loader, DataLoader)
    assert loader.batch_size == 4


def test_build_loader_default_params_dont_crash(tiny_image_folder):
    """Smoke test: building a loader with defaults must succeed."""
    ds = ImageFolder(str(tiny_image_folder))
    loader = build_loader(ds, num_workers=0)  # num_workers=0 avoids multiprocessing in tests
    assert loader is not None
