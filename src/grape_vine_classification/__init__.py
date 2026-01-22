from pathlib import Path
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PATH_DATA = PATH_DATA = PROJECT_ROOT / "data"
default_transform = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ]
)
