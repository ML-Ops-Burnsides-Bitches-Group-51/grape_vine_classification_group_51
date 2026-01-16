import torch
from src.grape_vine_classification.train_lightning import train
import os
from tests import PATH_MODEL, model_config_path

def test_train():
    train(model_config_path, model_name = "test_model.pth",logger = False)

    assert os.path.isfile(PATH_MODEL / "test_model.pth"), "No model saved"
