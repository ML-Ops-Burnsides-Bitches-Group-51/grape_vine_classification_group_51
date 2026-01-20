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
    with pytest.raises(ValueError, match="Expected input to have 4 dimensions"):
        model(torch.rand((1, 128, 128)))
    with pytest.raises(ValueError, match=re.escape("Expected sample to have size [1, 128, 128]")):
        model(torch.rand(1, 1, 128, 129))


@timeout_decorator.timeout(10)  # time out after a 10 seconds incase RuntimeError is not raised
def call_train_main(config_path: str = "", config: dict = None):
    train_main(config_path=config_path, config=config)


@timeout_decorator.timeout(10)  # time out after a 10 seconds incase RuntimeError is not raised
def call_sweep_main(config_path: str):
    sweep_main(config_path=config_path)


def test_error_on_invalid_config_path():
    dummy_path = "this_is_not_a_real_path/i_hope"
    with pytest.raises(RuntimeError, match="The config path is not valid"):
        call_train_main(dummy_path)
    with pytest.raises(RuntimeError, match="The config path is not valid"):
        call_sweep_main(dummy_path)


def test_validify_config():
    bad_config = model_config.copy()
    # if mandatory hyperparameter is not in config
    mandatory_hyperparameters = ["lr", "epochs", "patience", "optim"]
    for param in mandatory_hyperparameters:
        param_val = bad_config.pop(param)
        with pytest.raises(ValueError, match=f"Config does not contain: {param}"):
            call_train_main(config=bad_config)
        bad_config[param] = param_val
    # if optimzer is not supported
    bad_config["optim"] = "invalid_optim"
    with pytest.raises(ValueError, match="Specified optim is not supported: invalid_optim"):
        call_train_main(config=bad_config)
    # if optim = SGD but no momentum was specified
    bad_config["optim"] = "SGD"
    with pytest.raises(ValueError, match="Optim set to SGD, but config does not contain momentum"):
        call_train_main(config=bad_config)
