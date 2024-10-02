# -*- coding: utf-8 -*-
'''
MAT-Tools: Python Framework for Multiple Aspect Trajectory Data Mining

The present package offers a tool, to support the user in the task of data analysis of multiple aspect trajectories. It integrates into a unique framework for multiple aspects trajectories and in general for multidimensional sequence data mining methods.
Copyright (C) 2022, MIT license (this portion of code is subject to licensing from source project distribution)

Created on Dec, 2021
Copyright (C) 2022, License GPL Version 3 or superior (this portion of code is subject to licensing from source project distribution)

Authors:
    - Tarlis Portela
'''
# --------------------------------------------------------------------------------
import time
import pandas as pd
import numpy as np
from numpy import argmax

from tqdm.auto import tqdm

import itertools
# --------------------------------------------------------------------------------
from sklearn.neighbors import KNeighborsClassifier
# --------------------------------------------------------------------------------

from matclassification.methods.core import SimilarityClassifier

class TKNN(SimilarityClassifier):
    """
    A similarity-based classifier using k-Nearest Neighbors (KNN).

    This class extends the SimilarityClassifier to implement a similarity-based 
    classifier utilizing the k-Nearest Neighbors algorithm. It allows for 
    classification of trajectory data based on the distance between samples.

    Parameters:
    -----------
    k (list): 
        Number of neighbors to consider (default is [1, 3, 5]).
        
    weights (str): 
        Weight function used in prediction (default is 'distance').
        
    save_results (bool): 
        Flag indicating whether to save results (default is False).
            
    n_jobs : int, optional (default=-1)
        The number of parallel jobs to run for computation. -1 means using all processors.

    verbose : int, optional (default=2)
        Verbosity level. Higher values enable more detailed output during training and model creation.

    random_state : int, optional (default=42)
        Random seed used for reproducibility.

    filterwarnings : str, optional (default='ignore')
        Controls the filter for output warnings.

    Methods:
    --------
    fit(X, y): 
        Fit the SVC model to the training data.

    predict(X): 
        Predict class labels for the provided data.

    predict_proba(X): 
        Predict class probabilities for the provided data.

    evaluate(X_test, y_test): 
        Evaluate the model performance on the test data.
    """
    def __init__(self, 
                 k = [1, 3, 5], # Number of neighbors
                 weights = 'distance', # Weight function used in prediction ['uniform', 'distance']
                 
                 save_results=False,
                 n_jobs=1,
                 verbose=0,
                 random_state=42,
                 filterwarnings='ignore'):
        
        super().__init__('TKNN', save_results=save_results, n_jobs=n_jobs, verbose=verbose, random_state=random_state, filterwarnings=filterwarnings)
        
        self.add_config(k=k,
                        weights=weights)

        self.grid_search(k, weights)
        
    def create(self, config):

        k  = config[0]
        w  = config[1]
        
        # Initializing Model
        return KNeighborsClassifier(n_neighbors=k,
                                    weights=w,
                                    metric='precomputed', 
                                    n_jobs=self.config['n_jobs'])