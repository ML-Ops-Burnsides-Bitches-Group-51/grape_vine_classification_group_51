from pathlib import Path
import os
import torch
from torch.utils.data import random_split
from torchvision import datasets, transforms
import matplotlib.image
from grape_vine_classification import PATH_DATA
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

def preprocess():
    torch.manual_seed(1)

    # finding data directory from file directory
    raw_data_dir = PATH_DATA / "Grapevine_Leaves_Image_Dataset"
    processed_data_dir = PATH_DATA / "processed_dataset"

    # transform images
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Resize((128,128)),])

    # load images (each subfolder is interpreted as a class)
    full_dataset = datasets.ImageFolder(root=raw_data_dir, transform=transform)

    indices = list(range(len(full_dataset)))
    labels = [label for _, label in full_dataset.samples]

    # stratified split
    train_indices, test_indices = train_test_split(
        indices, test_size=0.2, stratify=labels, random_state=42
    )

    # create subsets
    train_dataset = Subset(full_dataset, train_indices)
    test_dataset = Subset(full_dataset, test_indices)

    # save test and train datasets
    os.makedirs(processed_data_dir, exist_ok=True)
    torch.save(train_dataset,f"{processed_data_dir}/train_data.pt")
    torch.save(test_dataset,f"{processed_data_dir}/test_data.pt")

    # save a sample image
    sample_img, _ = train_dataset[0]
    sample_img = sample_img[0].numpy()
    matplotlib.image.imsave(PATH_DATA / "processed_sample.png", sample_img)

if __name__ == "__main__":
    preprocess()
