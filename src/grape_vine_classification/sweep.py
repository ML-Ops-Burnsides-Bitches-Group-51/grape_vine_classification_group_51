import wandb
from grape_vine_classification.train_lightning import train
from pathlib import Path
import typer
import yaml
from pytorch_lightning.loggers import WandbLogger


def sweep_train():
    wandb.init()
    config = wandb.config
    logger = WandbLogger()
    train(config, logger=logger)


def main(config_path: str = "configs/sweep1_lightning.yaml", project_name: str = "sweeps"):
    path = Path(config_path)
    if path.exists():
        with open(path, "r") as f:
            sweep_config = yaml.safe_load(f)
    else:
        raise RuntimeError("The config path is not valid")

    sweep_id = wandb.sweep(sweep_config, project=project_name, entity="Burnsides_Bitches")

    wandb.agent(sweep_id, function=sweep_train)


if __name__ == "__main__":
    typer.run(main)
