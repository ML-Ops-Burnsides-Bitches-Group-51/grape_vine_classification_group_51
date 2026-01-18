from pathlib import Path
from grape_vine_classification.model_lightning import SimpleCNN
import matplotlib.pyplot as plt
import torch
import sys
from pytorch_lightning import Trainer 
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from grape_vine_classification import PATH_DATA
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset

import wandb
import yaml
import os
import typer

data_dir = PATH_DATA / "processed_dataset"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


# Hyperparamters
## Change this to load from config file

def train(config_path: str = "configs/experiment/exp1.yaml", model_name: str = "model.pth", logger: bool = True, sweep: bool = False) -> None:

    path = Path(config_path)
    config = {}
    if path.exists():
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        raise Warning("The config path is not valid")

    if "WANDB_SWEEP_ID" in os.environ:
        wandb.init()
        wandb.config.update(config, allow_val_change=True)
        config = wandb.config

 

    batch_size = config.get("batch_size")
    max_epochs = config.get("epochs")
    patience = config.get("patience")

    model = SimpleCNN(config)  # this is our LightningModule

    train_data = torch.load(data_dir / "train_data.pt", map_location=DEVICE)
    test_data = torch.load(data_dir / "test_data.pt", map_location=DEVICE)

    train_data = TensorDataset(train_data["images"],train_data["labels"])
    test_data = TensorDataset(test_data["images"],test_data["labels"])

    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=1)

    # Define trainer and train model
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=patience, mode="min"),
        ModelCheckpoint(dirpath="./models", monitor="val_loss", mode="min"),
    ]

    if logger:
        logger = pl.loggers.WandbLogger(project=config.get("project"),
                                            log_model="all")
        
    trainer = Trainer(logger=logger,max_epochs=max_epochs, callbacks=callbacks)
    trainer.fit(model, train_dataloader, test_dataloader)

    torch.save(model,"models/"+model_name)




if __name__ == "__main__":
    typer.run(train)

