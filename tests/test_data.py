from torch.utils.data import Dataset
import torch
import os.path
from tests import PATH_DATA
import pytest

processed_data_path = PATH_DATA / "processed_dataset"

@pytest.mark.skipif(not os.path.exists(processed_data_path), reason="Processed data folder not found")
def test_train_dataset():
    train_set = torch.load(processed_data_path / "train_data.pt")
    assert isinstance(train_set, Dataset), "Train dataset is wrong class"
    assert len(train_set) == 400, "Train dataset has wrong length"
    label_counts = [0,0,0,0,0]
    for i, (x, y) in enumerate(train_set):
        assert x.shape == (1, 128, 128), f"Image number {i} has wrong dimension"
        assert y in range(5), "Label is not an int between 0 and 4"
        label_counts[y] += 1
    for label in range(5):
        assert label_counts[label] == 80, f"Label ({label}) was not represented 80 times"

    # assert train_set.tensors[0].shape[1:] == torch.Size([1, 28, 28])
    # assert torch.unique(train_set.tensors[1]).tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# @pytest.mark.skipif(not os.path.exists(processed_data_path), reason="Processed data folder not found")
# def test_test_dataset():
#     _, test_set = corrupt_mnist()
#     assert isinstance(test_set, TensorDataset), "Test dataset wrong class"
#     assert len(test_set) == 5000, "Test dataset has wrong length"
#     assert test_set.tensors[0].shape[1:] == torch.Size([1, 28, 28])
#     assert torch.unique(test_set.tensors[1]).tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
