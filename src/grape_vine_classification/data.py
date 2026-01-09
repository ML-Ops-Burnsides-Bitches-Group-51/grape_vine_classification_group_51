from pathlib import Path
import os
import torch
from torch.utils.data import random_split
from torchvision import datasets, transforms
import matplotlib.image

def preprocess():
    torch.manual_seed(1)

    # finding data directory from file directory
    base_dir = Path(__file__).parent.parent.parent
    raw_data_dir = base_dir / "data" / "Grapevine_Leaves_Image_Dataset"
    processed_data_dir = base_dir / "data" / "processed_dataset"

    # transform images
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Resize((128,128)),])

    # load images (each subfolder is interpreted as a class)
    full_dataset = datasets.ImageFolder(root=raw_data_dir, transform=transform)

    # split dataset
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    # save test and train datasets
    os.makedirs(processed_data_dir, exist_ok=True)
    torch.save(train_dataset,f"{processed_data_dir}/train_data.pt")
    torch.save(test_dataset,f"{processed_data_dir}/test_data.pt")

    # save a sample image
    sample_img, _ = train_dataset[0]
    sample_img = sample_img[0].numpy()
    matplotlib.image.imsave(base_dir / "data" / "processed_sample.png", sample_img)

if __name__ == "__main__":
    preprocess()
