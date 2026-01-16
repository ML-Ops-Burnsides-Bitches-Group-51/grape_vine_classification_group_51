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

def train(config_path: str = "configs/exp1.yaml") -> None:
    # Load model and Data
    run = wandb.init(
        entity = "Burnsides_Bitches",
        project = "grape_classefier"
    )
    config = wandb.config

    path = Path(config_path)

    if not path.exists():
        print(f"Error: Config file {config_path} not found.")
        raise typer.Exit(code=1)
        

    with open(path, 'r') as f:
        local_config = yaml.safe_load(f)

    # If a sweeps is enabled then it will override the existing config (exp1.yaml)
    wandb.config.update(local_config, allow_val_change=True)
    config = wandb.config

    batch_size = config.get("batch_size")
    max_epochs = config.get("epochs")
    patience = config.get("patience")

    checkpoint_callback = ModelCheckpoint(
        dirpath="./models", 
        filename="best-checkpoint", 
        monitor="acc",               
        mode="max",                 
        save_top_k=1,                
    )

    model = SimpleCNN(config)  # this is our LightningModule
    train_data = torch.load(data_dir / "train_data.pt")
    test_data = torch.load(data_dir / "test_data.pt")
    
    


    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    # Define trainer and train model
    early_stopping_callback = EarlyStopping(
        monitor="acc", patience=patience, verbose=True, mode="max"
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath="./models", monitor="val_loss", mode="min"
    )

    wandb_logger = pl.loggers.WandbLogger(project="grape_vine_classification",
                                          log_model="all")
    trainer = Trainer(logger=wandb_logger,max_epochs=max_epochs, callbacks=[early_stopping_callback, checkpoint_callback])
    trainer.fit(model, train_dataloader, test_dataloader)


if __name__ == "__main__":
    typer.run(train)
    # train()
