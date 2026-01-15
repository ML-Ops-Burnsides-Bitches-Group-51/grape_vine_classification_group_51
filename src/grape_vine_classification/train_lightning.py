from pathlib import Path
from grape_vine_classification.model_lightning import SimpleCNN
import matplotlib.pyplot as plt
import torch
import typer
import sys
from pytorch_lightning import Trainer 
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from grape_vine_classification import PATH_DATA
import pytorch_lightning as pl

import wandb
import yaml

data_dir = PATH_DATA / "processed_dataset"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


# Hyperparamters
## Change this to load from config file
batch_size = 16
max_epochs = 10

def train(batch_size: int = 16, max_epochs: int = 10) -> None:
    # Load model and Data
    run = wandb.init(
        entity = "Burnsides_Bitches",
        project = "grape_classefier"
    )
    config = wandb.config

    if not config:
        # If config is empty (local run), load your exp1.yaml
        config_path = Path("configs/exp1.yaml")
        with open(config_path, 'r') as f:
            local_config = yaml.safe_load(f)
        
        # Update wandb with the local file's values
        wandb.config.update(local_config)
        config = wandb.config


    model = SimpleCNN(config)  # this is our LightningModule
    train_data = torch.load(data_dir / "train_data.pt")
    test_data = torch.load(data_dir / "test_data.pt")




    
    max_epochs = config.epochs


    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    # Define trainer and train model
    early_stopping_callback = EarlyStopping(
        monitor="val_loss", patience=3, verbose=True, mode="min"
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath="./models", monitor="val_loss", mode="min"
    )
    trainer = Trainer(logger=pl.loggers.WandbLogger(project="grape_vine_classification"),max_epochs=max_epochs, callbacks=[early_stopping_callback, checkpoint_callback])
    trainer.fit(model, train_dataloader, test_dataloader)


if __name__ == "__main__":
    #typer.run(train)
    train()
