from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PATH_DATA = PATH_DATA = PROJECT_ROOT / "data"
PATH_MODEL = PROJECT_ROOT / "model"

model_config_path = Path("configs/exp1.yaml")
with open(model_config_path, 'r') as f:
    model_config = yaml.safe_load(f)
