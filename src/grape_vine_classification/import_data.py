import kagglehub
import shutil
import os
from pathlib import Path

# 1. Download to default cache
cache_path = kagglehub.dataset_download("muratkokludataset/grapevine-leaves-image-dataset")

# 2. Define your target folder
base_dir = Path(__file__).parent.parent.parent
target_path = base_dir / "data"
os.makedirs(target_path, exist_ok=True)

# 3. Move files from cache to your folder
# Note: Use copy if you want to keep a backup in cache, or move to save space
for item in os.listdir(cache_path):
    s = os.path.join(cache_path, item)
    d = os.path.join(target_path, item)
    if os.path.isdir(s):
        shutil.copytree(s, d, dirs_exist_ok=True)
    else:
        shutil.copy2(s, d)

print(f"Data is now available in: {target_path}")