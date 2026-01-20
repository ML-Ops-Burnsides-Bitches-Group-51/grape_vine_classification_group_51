import torch
import os.path
from tests import PATH_DATA
import pytest
from torch.utils.data import TensorDataset

processed_data_path = PATH_DATA / "processed_dataset"


@pytest.mark.skipif(not os.path.exists(processed_data_path), reason="Processed data folder not found")
def test_train_dataset():
    train_set = torch.load(processed_data_path / "train_data.pt")
    assert isinstance(train_set, dict), "Train dataset is wrong class"
    assert len(train_set) == 2, "Train dataset has the wrong number of keys"

    label_counts = [0, 0, 0, 0, 0]
    train_set = TensorDataset(train_set["images"], train_set["labels"])

    for i, (x, y) in enumerate(train_set):
        assert x.shape == (1, 128, 128), f"Image number {i} has wrong dimension"
        assert y in range(5), f"Label number {i} is not an int between 0 and 4"
        label_counts[y] += 1
    for label in range(5):
        assert label_counts[label] == 80, f"Label ({label}) was not represented 80 times"


@pytest.mark.skipif(not os.path.exists(processed_data_path), reason="Processed data folder not found")
def test_test_dataset():
    test_set = torch.load(processed_data_path / "test_data.pt")
    assert isinstance(test_set, dict), "Test dataset is wrong class"
    assert len(test_set) == 2, "Test dataset has the wrong number of keys"

    label_counts = [0, 0, 0, 0, 0]
    test_set = TensorDataset(test_set["images"], test_set["labels"])

    for i, (x, y) in enumerate(test_set):
        assert x.shape == (1, 128, 128), f"Image number {i} has wrong dimension"
        assert y in range(5), f"Label number {i} is not an int between 0 and 4"
        label_counts[y] += 1
    for label in range(5):
        assert label_counts[label] == 20, f"Label ({label}) was not represented 20 times"
