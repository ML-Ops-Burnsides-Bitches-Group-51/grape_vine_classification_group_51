from pathlib import Path
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import typer
from torch.utils.data import Dataset
from torch import manual_seed


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
    T.Resize((224, 224)), # Temp not correct size
    T.ToTensor(),])
    # Load images
    train_images, train_target = [], []

    loader = DataLoader()
    for 

    
    # We must first split data in training and testing
    test_share = 0.2

    


    for i in range(6):
        train_images.append(torch.load(f"{raw_dir}/train_images_{i}.pt"))
        train_target.append(torch.load(f"{raw_dir}/train_target_{i}.pt"))
    train_images = torch.cat(train_images)
    train_target = torch.cat(train_target)

    test_images: torch.Tensor = torch.load(f"{raw_dir}/test_images.pt")
    test_target: torch.Tensor = torch.load(f"{raw_dir}/test_target.pt")

    train_images = train_images.unsqueeze(1).float()
    test_images = test_images.unsqueeze(1).float()
    train_target = train_target.long()
    test_target = test_target.long()

    # The images are now saved in the processed data folder

    train_images = normalize(train_images)
    test_images = normalize(test_images)

    torch.save(train_images, f"{processed_dir}/train_images.pt")
    torch.save(train_target, f"{processed_dir}/train_target.pt")

    torch.save(test_images, f"{processed_dir}/test_images.pt")
    torch.save(test_target, f"{processed_dir}/test_target.pt")
if __name__ == "__main__":
    typer.run(preprocess)
