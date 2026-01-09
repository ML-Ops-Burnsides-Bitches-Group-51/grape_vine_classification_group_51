# Grapevine leaf classification
## Project description
In this project we seek to classify grapevine leaves and to build an mlops pipeline around the model, to ensure reproducibility. We want to start with a simple convolution neural network (CNN) to check that everything runs as expected and then integrate torchvision into the framework, such that we are able to play around with different models and architechtures.

Note that we do not use pre-trained models, as we want to focus on making a complete pipeline including model training, and thus we want to integrate torchvision into the training framework. Furthermore we use pytorch lightning to ease the training, reduce boilerplate code and to easily switch between cpu and gpu. We will use Weights and Bias for experiment logging and hydra to write configuration files,
 

## Data description
There are five different species of grapes each with 100 unique images of a leaf. The original data is in RGB, but since all the leaves are similar in color, we have decided to convert the images to black and white to reduce the training time. Furthermore we have split the data into a training and a testing set. Note that since we only have 100 images of each species, we might have to augment our data, which we will do using AlbumentationsX.

The dataset can be found at : https://www.kaggle.com/datasets/muratkokludataset/grapevine-leaves-image-dataset

Note that the data is not included in the repository, and must thus be loaded using src/grape_vine_classification/import_data.py which downloads the data from kaggle into the data folder.

  
## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
