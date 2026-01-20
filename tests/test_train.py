from src.grape_vine_classification.train_lightning import train
import os
from tests import PATH_MODEL, model_config
import pytest

data_path = "data/processed_dataset/"


@pytest.mark.skipif(not os.path.exists(data_path), reason="Training data not available")
def test_train():
    train(model_config, output_model_name="test_model.pth", logger=False)

    assert os.path.isfile(PATH_MODEL / "test_model.pth"), "No model saved"
