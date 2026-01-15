from pathlib import Path
from grape_vine_classification.model import SimpleCNN
import matplotlib.pyplot as plt
import torch
import typer
import sys

from sklearn.metrics import RocCurveDisplay, accuracy_score, f1_score, precision_score, recall_score
import wandb

from grape_vine_classification import PATH_DATA



data_dir = PATH_DATA / "processed_dataset"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def get_optimizer(optmizer_name: str, model: SimpleCNN, lr: int):
    # Constructs an optmizer of the type defined by the config file
    if optmizer_name == "SGD":
        return torch.optim.SGD(model.parameters(),lr=lr,momentum=0.9) # Momentum is hardcoded, could be optmized for in sweep
    elif optmizer_name == "Adam":
        return torch.optim.Adam(model.parameters(),lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {optmizer_name}")
    
def train(lr: float = 1e-3, batch_size: int = 16, epochs: int = 100) -> None:
    """Train the model"""
    print("Training day and night")

    # Set the hyperparameters using wandb, necesary for sweeping
    run = wandb.init(
        entity = "Burnsides_Bitches",
        project = "grape_classefier"
    )
    config = wandb.config
    epochs = config.epochs

    # Currently hardcoded, should be a config parameter, that determines model and loss function
    model = SimpleCNN().to(DEVICE)
    optimizer = get_optimizer(config.optim,model,config.lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Data currently only works with the local file setup, needs to be changed for cloud
    train_data = torch.load(data_dir / "train_data.pt")
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=config.batch_size)
    

    statistics = {"train_loss": [], "train_accuracy": []}
    for epoch in range(epochs):
        model.train()
        preds, targets = [], []

        for i, (img, target) in enumerate(train_dataloader):
            img, target = img.to(DEVICE), target.to(DEVICE)

            optimizer.zero_grad()
            y_pred = model(img)
            loss = loss_fn(y_pred, target)
            loss.backward()
            optimizer.step()
            statistics["train_loss"].append(loss.item())

            wandb.log({"train_loss": loss.item()}) # This should log validation accuracy not trainning

            if i % 100 == 0:
                print(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")

        # preds = torch.cat(preds, 0)
        # targets = torch.cat(targets, 0)
    

    print("Training complete")

    torch.save(model.state_dict(), "models/model.pth")
    
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["train_loss"])
    axs[0].set_title("Train loss")
    axs[1].plot(statistics["train_accuracy"])
    axs[1].set_title("Train accuracy")
    fig.savefig("reports/figures/training_statistics.png")

def train():
    dataset = MyDataset("data/raw")
    model = Model()
    # add rest of your training code here


    artifact = wandb.Artifact(
        name="corrupt_mnist_model",
        type="model",
        description="A model trained to classify corrupt MNIST images",
      #  metadata={"accuracy": final_accuracy, "precision": final_precision, "recall": final_recall, "f1": final_f1},
    )
    artifact.add_file("model.pth") 
    run.log_artifact(artifact)


if __name__ == "__main__":
    train()
