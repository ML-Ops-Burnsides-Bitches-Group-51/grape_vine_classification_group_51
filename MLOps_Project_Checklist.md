# MLOps Project Checklist

## ✅ Week 1

- [x] Create a git repository (M5)
- [x] Make sure that all team members have write access to the GitHub repository (M5)
- [x] Create a dedicated environment for your project to keep track of your packages (M2)
- [x] Create the initial file structure using cookiecutter with an appropriate template (M6)
- [x] Fill out the data.py file such that it downloads whatever data you need and preprocesses it (if necessary) (M6)
- [x] Add a model to model.py and a training procedure to train.py and get that running (M6)
- [x] Fill out requirements.txt / requirements_dev.txt or keep pyproject.toml / uv.lock up to date (M2+M6)
- [x] Comply with good coding practices (PEP8) (M7)
- [ ] Document essential parts of your code (M7)
- [ ] Setup version control for your data or part of your data (M8)
- [ ] Add command line interfaces and project commands where it makes sense (M9)
- [ ] Construct one or multiple Dockerfiles for your code (M10)
- [ ] Build the Docker images locally and verify they work (M10)
- [x] Write one or multiple configuration files for experiments (M11)
- [ ] Use profiling to optimize your code (M12)
- [ ] Use logging to log important events in your code (M14)
- [x] Use Weights & Biases to log training progress and artifacts (M14) - Anton
- [x] Consider running a hyperparameter optimization sweep (M14)
- [x] Use PyTorch Lightning if applicable (M15) - Clara

[//]: # "- [ ] Use Hydra to load configurations and manage hyperparameters (M11)"
## ✅ Week 2

- [x] Write unit tests for data-related code (M16)
- [x] Write unit tests for model construction and/or training (M16)
- [x] Calculate code coverage (M16)
- [x] Setup continuous integration on GitHub (M17) - Anton
- [ ] Add caching and multi-OS/Python/PyTorch testing to CI (M17)
- [ ] Add linting to CI (M17)
- [ ] Add pre-commit hooks (M18)
- [ ] Add workflow triggered when data changes (M19)
- [x] Add workflow triggered when model registry changes (M19)
- [ ] Create GCP Bucket for data and link with DVC (M21)
- [ ] Create workflow to automatically build Docker images (M21)
- [ ] Train model on GCP using Engine or Vertex AI (M21)
- [ ] Create FastAPI inference application (M22) - Viktor
- [ ] Deploy model on GCP using Cloud Functions or Cloud Run (M23)
- [ ] Write API tests and integrate into CI (M24)
- [ ] Load test the application (M24)
- [ ] Create specialized deployment API using ONNX and/or BentoML (M25)
- [x] Create frontend for the API (M26)

## ✅ Week 3

- [ ] Check model robustness to data drift (M27)
- [ ] Collect input-output data from deployed application (M27)
- [ ] Deploy drift detection API (M27)
- [ ] Instrument API with system metrics (M28)
- [ ] Setup cloud monitoring (M28)
- [ ] Create alert systems in GCP (M28)
- [ ] Optimize data loading with distributed data loading if applicable (M29)
- [ ] Optimize training with distributed training if applicable (M30)
- [ ] Apply quantization / compilation / pruning for faster inference (M31)

## ⭐ Extra

- [ ] Write documentation for the application (M32)
- [ ] Publish documentation to GitHub Pages (M32)
- [ ] Revisit initial project description and evaluate outcome
- [ ] Create architectural diagram of MLOps pipeline
- [ ] Ensure all group members understand all project parts
- [ ] Upload all code to GitHub
