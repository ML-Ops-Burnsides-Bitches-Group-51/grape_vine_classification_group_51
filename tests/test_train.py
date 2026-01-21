from src.grape_vine_classification.train_lightning import train, get_model_type
import os
from tests import PATH_MODEL, model_config
import pytest
from pathlib import Path

data_path = "data/processed_dataset/"


@pytest.mark.skipif(not os.path.exists(data_path), reason="Training data not available")
def test_train():
    train(model_config, model_path = PATH_MODEL / "test_model.pth", logger=False)
    train(model_config, model_path = PATH_MODEL / "test_model.onnx", logger=False)

    assert os.path.isfile(PATH_MODEL / "test_model.pth"), "No .pth model saved"
    assert os.path.isfile(PATH_MODEL / "test_model.onnx"), "No .onnx model saved"

def test_validiy_model_type():
    with pytest.raises(ValueError, match="Unknown model type: .bad_suffix"):
        get_model_type(Path("this_is_not_a_real_path/i_hope.bad_suffix"))
    output = get_model_type(Path("this_is_not_a_real_path/i_hope.onnx"))
    assert output == "onnx"
    output = get_model_type(Path("this_is_not_a_real_path/i_hope.pth"))
    assert output == "pytorch"
    output = get_model_type(Path("this_is_not_a_real_path/i_hope.pt"))
    assert output == "pytorch"
