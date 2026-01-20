from grape_vine_classification.model import SimpleCNN
import matplotlib.pyplot as plt
import torch
from grape_vine_classification import PATH_DATA
from torch.utils.data import DataLoader, TensorDataset

data_dir = PATH_DATA / "processed_dataset"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def train(lr: float = 1e-3, batch_size: int = 16, epochs: int = 20, val_iter: int = 5) -> None:
    """Train the model"""
    print("Training day and night")
    print(f"{lr=}, {batch_size=}, {epochs=}, {val_iter=}")

    model = SimpleCNN().to(DEVICE)
    train_data = torch.load(data_dir / "train_data.pt", map_location=DEVICE)
    test_data = torch.load(data_dir / "test_data.pt", map_location=DEVICE)

    train_data = TensorDataset(train_data["images"], train_data["labels"])
    test_data = TensorDataset(test_data["images"], test_data["labels"])

    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    statistics = {"train_loss": ([], []), "train_accuracy": ([], []), "test_loss": ([], []), "test_accuracy": ([], [])}
    current_iteration = 0
    for epoch in range(epochs):
        model.train()
        for i, (img, target) in enumerate(train_dataloader):
            current_iteration += len(target)
            img, target = img.to(DEVICE), target.to(DEVICE)

            optimizer.zero_grad()
            y_pred = model(img)
            loss = loss_fn(y_pred, target)
            loss.backward()
            optimizer.step()
            statistics["train_loss"][0].append(current_iteration)
            statistics["train_loss"][1].append(loss.item())

            accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()
            statistics["train_accuracy"][0].append(current_iteration)
            statistics["train_accuracy"][1].append(accuracy)

            if i % 100 == 0:
                print(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")

        if (epoch % val_iter == 0 and val_iter) or (epoch + 1 == epochs):
            model.eval()
            val_cum_loss = torch.zeros(1)
            val_cum_accuracy = torch.zeros(1)
            for i, (img, target) in enumerate(test_dataloader):
                img, target = img.to(DEVICE), target.to(DEVICE)
                with torch.no_grad():
                    y_pred = model(img)
                val_loss = loss_fn(y_pred, target)
                val_cum_loss += val_loss
                val_accuracy = (y_pred.argmax(dim=1) == target).float().mean()
                val_cum_accuracy += val_accuracy
            val_loss_avg = val_cum_loss.item() / len(test_dataloader)
            val_accuracy_avg = val_cum_accuracy.item() / len(test_dataloader)
            statistics["test_loss"][0].append(current_iteration)
            statistics["test_loss"][1].append(val_loss_avg)
            statistics["test_accuracy"][0].append(current_iteration)
            statistics["test_accuracy"][1].append(val_accuracy_avg)
            print(f"Epoch {epoch}, validation loss: {val_loss_avg}")
            print(f"Epoch {epoch}, validation acc: {val_accuracy_avg}")

    print("Training complete")
    torch.save(model.state_dict(), "models/model.pth")
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["train_loss"][0], statistics["train_loss"][1])
    axs[0].plot(statistics["test_loss"][0], statistics["test_loss"][1])
    axs[0].set_title("Loss")
    axs[1].plot(statistics["train_accuracy"][0], statistics["train_accuracy"][1])
    axs[1].plot(statistics["test_accuracy"][0], statistics["test_accuracy"][1])
    axs[1].set_title("Accuracy")
    fig.savefig("reports/figures/training_statistics.png")


if __name__ == "__main__":
    train()
