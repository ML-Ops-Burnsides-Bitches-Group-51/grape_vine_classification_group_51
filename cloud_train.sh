#!/bin/bash

# --- 0. Setup Environment ---

export PATH=$PATH:/usr/bin:/usr/local/bin

# Install gcsfuse if it's missing (common on Ubuntu-based DL images)
if ! command -v gcsfuse &> /dev/null; then
    export DISTRIBUTION_CORE=$(lsb_release -c -s)
    echo "deb https://packages.cloud.google.com/apt gcsfuse-$DISTRIBUTION_CORE main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
    sudo apt-get update
    sudo apt-get install -y gcsfuse
fi

# 1. Create the mount points on the VM
mkdir -p /mnt/data
mkdir -p /mnt/models

# 2. Mount the BUCKETS (Use the Bucket Name, not a path)
# Replace 'grapevine_data' with your EXACT bucket name from the GCP console
gcsfuse --implicit-dirs grapevine_data /mnt/data
gcsfuse --implicit-dirs models_grape_gang /mnt/models

# 3. Pull the Docker image
docker pull europe-west1-docker.pkg.dev/grapevine-gang/grape-functions/trainer:v1

# 4. Run the container
# FORMAT: -v [FOLDER_ON_VM]:[FOLDER_IN_CONTAINER]
docker run --rm \
  --gpus all \
  -e WANDB_API_KEY="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" \
  -v /mnt/data:/data:ro \
  -v /mnt/models:/models \
  europe-west1-docker.pkg.dev/grapevine-gang/grape-functions/trainer:v1 \
  --config-path configs/experiment/exp1.yaml \
  --data-path /data/processed_dataset \
  --model-path /models/final_trained_model.pth