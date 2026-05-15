from __future__ import annotations
import pytest
import torch
import torch.nn as nn

from src.classifier.model import create_model

@pytest.mark.slow
def test_create_model_returns_module():
    model = create_model()
    assert isinstance(model, nn.Module)


@pytest.mark.slow
def test_create_model_default_output_shape():
    model = create_model()
    model.eval()

    batch = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(batch)

    assert out.shape == (2, 152), f"expected (2, 152), got {tuple(out.shape)}"


@pytest.mark.slow
def test_create_model_custom_num_classes():
    model = create_model(num_classes=10)
    model.eval()
    out = model(torch.randn(1, 3, 224, 224))
    assert out.shape == (1, 10)


@pytest.mark.slow
def test_create_model_accepts_alternative_backbone():
    model = create_model(model_name="resnet18", num_classes=152)
    model.eval()
    out = model(torch.randn(1, 3, 224, 224))
    assert out.shape == (1, 152)
