from pathlib import Path
import os
import torch
from torch.utils.data import random_split, DataLoader
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
        transforms.Resize((128,128)),
        transforms.ToTensor(),])

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

    loader_train = DataLoader(train_dataset, batch_size=len(train_dataset))
    train_data, train_labels = next(iter(loader_train))

    loader_test = DataLoader(test_dataset,batch_size= len(test_dataset)) # This should be 1
    test_data, test_labels = next(iter(loader_test))

    os.makedirs(processed_data_dir, exist_ok=True)
    
    # We save a dictionary. This is environment-agnostic.
    torch.save({"images": train_data, "labels": train_labels}, f"{processed_data_dir}/train_data.pt")
    torch.save({"images": test_data, "labels": test_labels}, f"{processed_data_dir}/test_data.pt")

    # save a sample image (using the tensor we just extracted)
    sample_img = train_data[0][0].numpy()
    matplotlib.image.imsave(PATH_DATA / "processed_sample.png", sample_img, cmap='gray')

if __name__ == "__main__":
    preprocess()
