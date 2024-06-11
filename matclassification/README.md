# MAT-classification: Analysis and Classification methods for Multiple Aspect Trajectory Data Mining \[MAT-Tools Framework\]
---

\[[Publication](#)\] \[[citation.bib](citation.bib)\] \[[GitHub](https://github.com/mat-analysis/mat-classification)\] \[[PyPi](https://pypi.org/project/mat-classification/)\]


The present package offers a tool, to support the user in the task of classification of multiple aspect trajectories. It integrates into a unique framework for multiple aspects trajectories and in general for multidimensional sequence data mining methods.

Created on Dec, 2023
Copyright (C) 2023, License GPL Version 3 or superior (see LICENSE file)


### Installation

Install directly from PyPi repository, or, download from github. (python >= 3.7 required)

```bash
    pip3 install mat-classification
```

### Getting Started

On how to use this package, see [MAT-classification-Tutorial.ipynb](https://github.com/mat-analysis/mat-classification/blob/main/MAT-classification-Tutorial.ipynb)

### Available Classifiers:

#### Movelet-Based:

* **MMLP (Movelet)**: Movelet Multilayer-Perceptron (MLP) with movelets features. The models were implemented using the Python language, with the keras, fully-connected hidden layer of 100 units, Dropout Layer with dropout rate of 0.5, learning rate of 10−3 and softmax activation function in the Output Layer. Adam Optimization is used to avoid the categorical cross entropy loss, with 200 of batch size, and a total of 200 epochs per training. \[[REFERENCE](https://doi.org/10.1007/s10618-020-00676-x)\]
* **MRF (Movelet)**: Movelet Random Forest (RF) with movelets features, that consists of an ensemble of 300 decision trees. The models were implemented using the Python language, with the keras. \[[REFERENCE](https://doi.org/10.1007/s10618-020-00676-x)\]
* **MSVN (Movelet)**: Movelet Support Vector Machine (SVM) with movelets features. The models were implemented using the Python language, with the keras, linear kernel and default structure. Other structure details are default settings. \[[REFERENCE](https://doi.org/10.1007/s10618-020-00676-x)\]

#### Feature-Based:
* **POI-S**: Frequency-based method to extract features of trajectory datasets (TF-IDF approach), the method runs one dimension at a time (or more if concatenated). The models were implemented using the Python language, with the keras. \[[REFERENCE](https://doi.org/10.1145/3341105.3374045)\]

#### Trajectory-Based:

* **MARC**: Uses word embeddings for trajectory classification. It encapsulates all trajectory dimensions: space, time and semantics, and uses them as input to a neural network classifier, and use the geoHash on the spatial dimension, combined with others. The models were implemented using the Python language, with the keras. \[[REFERENCE](https://doi.org/10.1080/13658816.2019.1707835)\]
* **TRF**: Random Forest for trajectory data (TRF). Find the optimal set of hyperparameters for each model, applying the grid-search technique: varying number of trees (ne), the maximum number of features to consider at every split (mf), the maximum number of levels in a tree (md), the minimum number of samples required to split a node (mss), the minimum number of samples required at each leaf node (msl), and finally, the method of selecting samples for training each tree (bs). \[[REFERENCE](http://dx.doi.org/10.5220/0010227906640671)\]
* **TXGBost**: Find the optimal set of hyperparameters for each model, applying the grid-search technique:  number of estimators (ne), the maximum depth of a tree (md), the learning rate (lr), the gamma (gm), the fraction of observations to be randomly samples for each tree (ss), the sub sample ratio of columns when constructing each tree (cst), the regularization parameters (l1) and (l2). \[[REFERENCE](http://dx.doi.org/10.5220/0010227906640671)\]
* **BiTuler**: Find the optimal set of hyperparameters for each model, applying the grid-search technique: keeps 64 as the batch size and 0.001 as the learning rate and vary the units (un) of the recurrent layer, the embedding size to each attribute (es) and the dropout (dp). \[[REFERENCE](http://dx.doi.org/10.5220/0010227906640671)\]
* **Tulvae**: Find the optimal set of hyperparameters for each model, applying the grid-search technique: keeps 64 as the batch size and 0.001 as the learning rate and vary the units (un) of the recurrent layer, the embedding size to each attribute (es), the dropout (dp), and latent variable (z). \[[REFERENCE](http://dx.doi.org/10.5220/0010227906640671)\]
* **DeepeST**: DeepeST employs a Recurrent Neural Network (RNN), both LSTM and Bidirectional LSTM (BLSTM). Find the optimal set of hyperparameters for each model, applying the grid-search technique: keeps 64 as the batch size and 0.001 as the learning rate and vary the units (un) of the recurrent layer, the embedding size to each attribute (es) and the dropout (dp). \[[REFERENCE](http://dx.doi.org/10.5220/0010227906640671)\]

#### Available Scripts (TODO update):

By installing the package the following python scripts will be installed for use in system command line tools:

* `MAT-TC.py`: Script to run classifiers on trajectory datasets, for details type: `MAT-TC.py --help`;
* `MAT-MC.py`: Script to run **movelet-based** classifiers on trajectory datasets, for details type: `MAT-MC.py --help`;
* `POIS-TC.py`: Script to run POI-F/POI-S classifiers on the methods feature matrix, for details type: `POIS-TC.py --help`;
* `MARC.py`: Script to run MARC classifier on trajectory datasets, for details type: `MARC.py --help`.

One script for running the **POI-F/POI-S** method:

* `POIS.py`: Script to run POI-F/POI-S feature extraction methods (`poi`, `npoi`, and `wnpoi`), for details type: `POIS.py --help`.

And one script for merging movelet resulting matrices:

* `MAT-MergeDatasets.py`: Script to join all class train.csv and test.csv of movelets for using as input into a classifier, for details type: `MAT-MergeDatasets.py --help`.

### Citing

If you use `mat-classification` please cite the following paper:

**TODO**

Bibtex:
```bash
@inproceedings{...}
```

### Collaborate with us

Any contribution is welcome. This is an active project and if you would like to include your code, feel free to fork the project, open an issue and contact us.

Feel free to contribute in any form, such as scientific publications referencing this package, teaching material and workshop videos.

### Related packages

This package is part of _MAT-Tools Framework_ for Multiple Aspect Trajectory Data Mining, check the guide project:

- **[mat-tools](https://github.com/mat-analysis/mat-tools)**: Reference guide for MAT-Tools Framework repositories

### Change Log

This is a package under construction, see [CHANGELOG.md](./CHANGELOG.md)
