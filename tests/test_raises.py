import pytest
from src.grape_vine_classification.model_lightning import SimpleCNN
from tests import model_config
import torch
import re

def test_error_on_wrong_shape():
    model = SimpleCNN(model_config)
    with pytest.raises(ValueError, match = "Expected input to have 4 dimensions"):
        model(torch.rand((1,128,128)))
    with pytest.raises(ValueError, match = re.escape("Expected sample to have size [1, 128, 128]")):
        model(torch.rand(1,1,128,129)) 