import torch
from src.grape_vine_classification.train_lightning import train
import os
from tests import PATH_MODEL

def test_train():
    train(config_path= "",model_name="test_model.pth")

    assert os.path.isfile(PATH_MODEL / "test_model.pth"), "No model saved"
