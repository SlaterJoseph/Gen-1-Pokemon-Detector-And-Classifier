from __future__ import annotations
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2
from src.classifier.transforms import train_transform, eval_transform


def _dummy_pil_image(size: int = 256) -> Image.Image:
    arr = np.random.randint(0, 256, (size, size, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def test_train_transform_returns_compose():
    assert isinstance(train_transform(), v2.Compose)


def test_eval_transform_returns_compose():
    assert isinstance(eval_transform(), v2.Compose)


def test_train_transform_output_shape_and_dtype():
    img = _dummy_pil_image()
    out = train_transform()(img)
    assert out.shape == (3, 224, 224), f"expected (3, 224, 224), got {tuple(out.shape)}"
    assert out.dtype == torch.float32


def test_eval_transform_output_shape_and_dtype():
    img = _dummy_pil_image()
    out = eval_transform()(img)
    assert out.shape == (3, 224, 224)
    assert out.dtype == torch.float32


def test_train_transform_outputs_normalized_range():
    img = _dummy_pil_image()
    out = train_transform()(img)
    assert out.min().item() >= -3.0, f"min out of expected range: {out.min().item()}"
    assert out.max().item() <= 3.0, f"max out of expected range: {out.max().item()}"


def test_eval_transform_outputs_normalized_range():
    img = _dummy_pil_image()
    out = eval_transform()(img)
    assert out.min().item() >= -3.0
    assert out.max().item() <= 3.0


def test_eval_transform_is_deterministic():
    img = _dummy_pil_image()
    tf = eval_transform()
    a = tf(img)
    b = tf(img)
    assert torch.equal(a, b), "eval_transform must be deterministic"


def test_train_transform_is_stochastic():
    img = _dummy_pil_image()
    tf = train_transform()
    a = tf(img)
    b = tf(img)
    assert not torch.equal(a, b), "train_transform must include random augmentation"
