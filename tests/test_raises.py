import pytest
from src.grape_vine_classification.model_lightning import SimpleCNN
from src.grape_vine_classification.train_lightning import main as train_main
from src.grape_vine_classification.sweep import main as sweep_main
from tests import model_config
import torch
import re
import timeout_decorator

def test_error_on_wrong_shape():
    model = SimpleCNN(model_config)
    with pytest.raises(ValueError, match = "Expected input to have 4 dimensions"):
        model(torch.rand((1,128,128)))
    with pytest.raises(ValueError, match = re.escape("Expected sample to have size [1, 128, 128]")):
        model(torch.rand(1,1,128,129)) 

@timeout_decorator.timeout(10) # time out after a 10 seconds incase RuntimeError is not raised
def call_train_main(config_path):
    train_main(config_path = config_path)

@timeout_decorator.timeout(10) # time out after a 10 seconds incase RuntimeError is not raised
def call_sweep_main(config_path):
    sweep_main(config_path = config_path)

def test_error_on_invalid_config_path():
    dummy_path = "this_is_not_a_real_path/i_hope"
    with pytest.raises(RuntimeError, match = "The config path is not valid"):
        call_train_main(dummy_path)
    with pytest.raises(RuntimeError, match = "The config path is not valid"):
        call_sweep_main(dummy_path)