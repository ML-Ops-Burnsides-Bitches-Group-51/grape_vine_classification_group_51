from pathlib import Path
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import typer
from torch.utils.data import Dataset, random_split
from torch import manual_seed, save
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os

class MyDataset(Dataset):
    """My custom dataset."""

    def __init__(self, data_path: Path) -> None:
        self.data_path = data_path

    def __len__(self) -> int:
        """Return the length of the dataset."""

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data and save it to the output folder."""



def preprocess(raw_dir, processed_dir):
    manual_seed(1)
    # Preprocess the data

    transform = T.Compose([
    transforms.Grayscale(num_output_channels=1),
    T.ToTensor(),])
    # Load images
    # base_dir

    full_dataset = datasets.ImageFolder(root=raw_dir, transform=transform)

    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size

    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])


    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
    save(train_dataset,f"{processed_dir}/train_data.pt")
    save(test_dataset,f"{processed_dir}/test_data.pt")



if __name__ == "__main__":
    preprocess("data/raw_data//Grapevine_Leaves_Image_Dataset","data/proccesed_data/")
