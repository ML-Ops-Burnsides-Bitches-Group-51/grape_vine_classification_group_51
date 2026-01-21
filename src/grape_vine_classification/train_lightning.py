from pathlib import Path
from grape_vine_classification.model_lightning import SimpleCNN
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from grape_vine_classification import PATH_DATA, PROJECT_ROOT
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, TensorDataset
import yaml
import typer

data_dir = PATH_DATA / "processed_dataset"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def validify_config(config: dict):
    # {'batch_size': 16, 'epochs': 5, 'lr': 0.001, 'momentum': 0.9, 'optim': 'Adam', 'patience': 3}
    mandatory_hyperparameters = ["lr", "epochs", "patience", "optim"]
    for param in mandatory_hyperparameters:
        if param not in config:
            raise ValueError(f"Config does not contain: {param}")
    if "optim" in config:
        optim = config["optim"]
        if optim not in ["Adam", "SGD"]:
            raise ValueError(f"Specified optim is not supported: {optim}")
        if (optim == "SGD") and ("momentum" not in config):
            raise ValueError("Optim set to SGD, but config does not contain momentum")


def train(config: dict = {}, logger = False, model_path = PROJECT_ROOT / "models" / "model.pth", data_path = PATH_DATA / "processed_dataset") -> None:
    validify_config(config)
    batch_size = config["batch_size"]
    max_epochs = config["epochs"]
    patience = config["patience"]

    model = SimpleCNN(config)  # this is our LightningModule

    train_data = torch.load(data_dir / "train_data.pt", map_location=DEVICE)
    test_data = torch.load(data_dir / "test_data.pt", map_location=DEVICE)

    train_data = TensorDataset(train_data["images"], train_data["labels"])
    test_data = TensorDataset(test_data["images"], test_data["labels"])

    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=1)

    # Define trainer and train model
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=patience, mode="min"),
        ModelCheckpoint(dirpath="./models", monitor="val_loss", mode="min"),
    ]

    trainer = Trainer(logger=logger, max_epochs=max_epochs, callbacks=callbacks)
    trainer.fit(model, train_dataloader, test_dataloader)

    torch.save(model, model_path)


def main(config_path: str = "configs/experiment/exp1.yaml",
         config = None, data_path: Path = PATH_DATA / "processed_dataset",
         model_path: Path = PROJECT_ROOT / "models" / "model.pth"):
    if not config:
        path = Path(config_path)
        if path.exists():
            with open(path, "r") as f:
                config = yaml.safe_load(f)
        else:
            raise RuntimeError("The config path is not valid")
    logger = WandbLogger(project="runs", entity="Burnsides_Bitches", config=config)
    train(config, logger = logger, data_path = data_path, model_path = model_path)


if __name__ == "__main__":
    typer.run(main)
