Product subscription
==============================

Project n°1 - Yotta Academy 2021 - Machine Learning Engineer bootcamp.

Authors: Olivier COLLIER & Jérémie KOSTER

# Context

The goal of this academic project is to make a machine learning model that predicts whether a bank client will subscribe to a product or not.

# Pre-requisites

- Python 3.8 (see the installation guide on [GitHub](https://github.com/pyenv/pyenv))
- Poetry (see the installation [guide](https://python-poetry.org/docs/#installation))
- Git (see the installation [guide](https://git-scm.com/book/fr/v2/D%C3%A9marrage-rapide-Installation-de-Git))

**Note**: some dependencies may not be installed with Python 3.9

# Installation

To install the project on your computer:

    $ git clone https://gitlab.com/yotta-academy/mle-bootcamp/projects/ml-project/winter-2021/productsubscription-oc-jk.git
    $ cd productsubscription-oc-jk

To install the required dependencies and activate the virtual environment:

    $ source activate.sh

# How to use

## Train

> Place the client dataset (`data.csv`) and the socio-economic dataset (`socio_eco.csv`) in the `data/raw/train` directory prior to the training.

    $ poetry run train

The model is then saved in the `models/` folder.

## Predict

> Place the client dataset (`data.csv`) and the socio-economic dataset (`socio_eco.csv`) in the `data/raw/predict` directory prior to the predict.

    $ poetry run predict

The predictions are stored in the `data/predictions/` folder.
