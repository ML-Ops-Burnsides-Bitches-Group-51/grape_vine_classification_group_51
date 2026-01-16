from src.grape_vine_classification.model_lightning import SimpleCNN
import torch
import yaml
from pathlib import Path
from tests import model_config

def test_model_output_shape():
    model = SimpleCNN(model_config)
    x = torch.randn(1, 1, 128, 128)
    y = model(x)
    assert y.shape == (1, 5)