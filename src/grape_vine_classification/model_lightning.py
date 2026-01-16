from torch import nn, optim
import torch
from pytorch_lightning import LightningModule

class SimpleCNN(LightningModule):
    """My awesome model."""

    def __init__(self, config) -> None:
        self.config = config
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(3, 3),

            nn.Conv2d(16, 32, 3, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(3, 3),

            nn.Conv2d(32, 64, 3, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 3),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(576, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 5)
        )
        self.criterium = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # value errors for unit testing
        if x.ndim != 4:
            raise ValueError("Expected input to have 4 dimensions")
        if x.shape[1:] != torch.Size([1, 128, 128]):
            raise ValueError("Expected sample to have size [1, 128, 128]")

        return self.classifier(self.backbone(x))

    def training_step(self, batch, batch_idx: int) -> float:
        data, target = batch
        preds = self(data)
        loss = self.criterium(preds, target)
        return loss
   
    def validation_step(self, batch, batch_idx: int) -> None:
        # this is the validation loop
        data, target = batch
        preds = self(data)
        val_loss = self.criterium(preds, target)
        acc = (target == preds.argmax(dim=-1)).float().mean()
        self.log("val_loss", val_loss, on_epoch=True)
        self.log("acc", acc, on_epoch=True)

    def configure_optimizers(self):
        if self.config["optim"] == "Adam":
            return optim.Adam(self.parameters(), lr = self.config["lr"])
        elif self.config["optim"] == "SGD":
            return optim.SGD(self.parameters(), lr = self.config["lr"], momentum = self.config["momentum"])
    
   


if __name__ == "__main__":
    model = SimpleCNN()
    print(f"Model architecture: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    dummy_input = torch.randn(1, 1, 128, 128)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
