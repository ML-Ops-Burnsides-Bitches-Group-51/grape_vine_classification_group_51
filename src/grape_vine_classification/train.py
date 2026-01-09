from pathlib import Path
from grape_vine_classification.model import SimpleCNN
import matplotlib.pyplot as plt
import torch
import typer
import sys

data_dir = Path(__file__).parent.parent.parent / "data" / "processed_dataset"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def train(lr: float = 1e-3, batch_size: int = 16, epochs: int = 100) -> None:
    """Train the model"""
    print("Training day and night")
    print(f"{lr=}, {batch_size=}, {epochs=}")

    model = SimpleCNN().to(DEVICE)
    train_data = torch.load(data_dir / "train_data.pt")

    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    statistics = {"train_loss": [], "train_accuracy": []}
    for epoch in range(epochs):
        model.train()
        for i, (img, target) in enumerate(train_dataloader):
            img, target = img.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            y_pred = model(img)
            loss = loss_fn(y_pred, target)
            loss.backward()
            optimizer.step()
            statistics["train_loss"].append(loss.item())

            accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()
            statistics["train_accuracy"].append(accuracy)

            if i % 100 == 0:
                print(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")

    print("Training complete")
    torch.save(model.state_dict(), "models/model.pth")
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["train_loss"])
    axs[0].set_title("Train loss")
    axs[1].plot(statistics["train_accuracy"])
    axs[1].set_title("Train accuracy")
    fig.savefig("reports/figures/training_statistics.png")


if __name__ == "__main__":
    #typer.run(train)
    train()
