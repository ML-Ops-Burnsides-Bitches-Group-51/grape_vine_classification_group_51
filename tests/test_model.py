from src.grape_vine_classification.model_lightning import SimpleCNN
import torch
import yaml
from pathlib import Path

config_path = Path("configs/exp1.yaml")
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

def test_model_output_shape():
    model = SimpleCNN(config)
    x = torch.randn(1, 1, 128, 128)
    y = model(x)
    assert y.shape == (1, 5)