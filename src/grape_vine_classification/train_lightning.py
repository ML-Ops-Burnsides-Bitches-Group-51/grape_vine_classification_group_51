from pathlib import Path
from grape_vine_classification.model_lightning import SimpleCNN
import matplotlib.pyplot as plt
import torch
import typer
import sys
from pytorch_lightning import Trainer 
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from grape_vine_classification import PATH_DATA

data_dir = PATH_DATA / "processed_dataset"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


# Hyperparamters
## Change this to load from config file
batch_size = 16
max_epochs = 10

# Load model and Data
model = SimpleCNN()  # this is our LightningModule
train_data = torch.load(data_dir / "train_data.pt")
test_data = torch.load(data_dir / "test_data.pt")
train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

# Define trainer and train model
early_stopping_callback = EarlyStopping(
    monitor="val_loss", patience=3, verbose=True, mode="min"
)
checkpoint_callback = ModelCheckpoint(
    dirpath="./models", monitor="val_loss", mode="min"
)
trainer = Trainer(max_epochs=max_epochs, callbacks=[early_stopping_callback, checkpoint_callback])
trainer.fit(model, train_dataloader, test_dataloader)

"""
def train(lr: float = 1e-3, batch_size: int = 16, epochs: int = 100) -> None:
    # Train the model
    print("Training day and night")
    print(f"{lr=}, {batch_size=}, {epochs=}, {val_iter=}")

    model = SimpleCNN().to(DEVICE)
    train_data = torch.load(data_dir / "train_data.pt")
    test_data = torch.load(data_dir / "test_data.pt")

    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
    test_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)

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
            print(f"Validation loss, epoch {epoch}, loss: {val_loss_avg}")
            print(f"Validation accuracy, epoch {epoch}, loss: {val_accuracy_avg}")

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
"""

#if __name__ == "__main__":
    #typer.run(train)
    #train()
