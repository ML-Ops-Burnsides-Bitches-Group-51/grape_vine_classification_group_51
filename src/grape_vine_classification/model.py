from torch import nn
import torch

class SimpleCNN(nn.Module):
    """My awesome model."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(576, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 3, 3)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 3, 3)
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, 3, 3)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc1(x)


if __name__ == "__main__":
    model = SimpleCNN()
    print(f"Model architecture: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    dummy_input = torch.randn(1, 1, 512, 512)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
