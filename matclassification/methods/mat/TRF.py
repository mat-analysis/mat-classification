# -*- coding: utf-8 -*-
'''
MAT-Tools: Python Framework for Multiple Aspect Trajectory Data Mining

The present package offers a tool, to support the user in the task of data analysis of multiple aspect trajectories. It integrates into a unique framework for multiple aspects trajectories and in general for multidimensional sequence data mining methods.
Copyright (C) 2022, MIT license (this portion of code is subject to licensing from source project distribution)

Created on Dec, 2021
Copyright (C) 2022, License GPL Version 3 or superior (this portion of code is subject to licensing from source project distribution)

Authors:
    - Tarlis Portela
    - Original source:
        - Nicksson C. A. de Freitas, 
        - Ticiana L. Coelho da Silva, 
        - Jose António Fernandes de Macêdo, 
        - Leopoldo Melo Junior, 
        - Matheus Gomes Cordeiro
    - Adapted from: https://github.com/nickssonfreitas/ICAART2021
'''
# --------------------------------------------------------------------------------
import time
import pandas as pd
import numpy as np
from numpy import argmax

from tqdm.auto import tqdm

import itertools
# --------------------------------------------------------------------------------
from sklearn.ensemble import RandomForestClassifier
# --------------------------------------------------------------------------------

from matclassification.methods.core import THSClassifier, compute_acc_acc5_f1_prec_rec

class TRF(THSClassifier):
    """
    TRF: Trajectory Random Forest Classifier

    The `TRF` class is a Random Forest trajectory-based classifier specifically designed for trajectory classification tasks. It provides tunable hyperparameters and performs grid search to find the optimal model configuration. 
    #TODO: It supports parallelization, result saving, and allows for a flexible and efficient approach to handling complex trajectory data.

    Parameters
    ----------
    n_estimators : list, default=[200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
        Number of trees in the forest.
    max_depth : list, default=[20, 30, 40]
        Maximum depth of each tree.
    min_samples_split : list, default=[2, 5, 10]
        Minimum number of samples required to split an internal node.
    min_samples_leaf : list, default=[1, 2, 4]
        Minimum number of samples required to be at a leaf node.
    max_features : list, default=['sqrt', 'log2']
        Number of features to consider when looking for the best split.
    bootstrap : list, default=[True, False]
        Method of selecting samples for training each tree (whether to use bootstrap sampling).
    save_results : bool, default=False
        Option to save the results of the classifier.
    n_jobs : int, default=-1
        Number of jobs to run in parallel. -1 means using all processors.
    verbose : int, default=0
        Controls the verbosity during model training.
    random_state : int, default=42
        Controls the randomness of the estimator.
    filterwarnings : str, default='ignore'
        Whether to suppress or display warnings during model training and evaluation.

    Methods
    -------
    create(config):
        Initializes the Random Forest classifier with the given configuration.
    fit(X_train, y_train, X_val, y_val, config=None):
        Trains the Random Forest classifier on the provided training data.
    predict(X_test, y_test):
        Predicts class probabilities and the most likely class for the test data.
    """
    
    def __init__(self, 
                 n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)], # Number of trees in random forest
                 max_depth = [int(x) for x in np.linspace(20, 40, num = 3)], # Maximum number of levels in tree
                 min_samples_split =  [2, 5, 10], # Minimum number of samples required to split a node
                 min_samples_leaf =  [1, 2, 4], # Minimum number of samples required at each leaf node
                 max_features= ['sqrt', 'log2'], #['auto', 'sqrt'] # Number of features to consider at every split 
                 # Tarlis: max_features 'auto' is deprecated, replaced ['auto', 'sqrt'] with: ['sqrt', 'log2'] ?
                 bootstrap =  [True, False], # Method of selecting samples for training each tree
                 
                 save_results=False,
                 n_jobs=-1,
                 verbose=0,
                 random_state=42,
                 filterwarnings='ignore'):
        
        super().__init__('TRF', save_results=save_results, n_jobs=n_jobs, verbose=verbose, random_state=random_state, filterwarnings=filterwarnings)
        
        self.add_config(n_estimators=n_estimators, 
                        max_depth=max_depth, 
                        min_samples_split=min_samples_split, 
                        min_samples_leaf=min_samples_leaf, 
                        max_features=max_features, 
                        bootstrap=bootstrap)

        self.grid_search(n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, bootstrap)
#        self.grid = list(itertools.product(n_estimators, max_depth, min_samples_split, 
#                                           min_samples_leaf, max_features, bootstrap))
        
        self.model = None
        
    def create(self, config):

        ne  = config[0]
        md  = config[1]
        mss = config[2]
        msl = config[3]
        mf  = config[4]
        bs  = config[5]
        
        # Initializing Model
        return RandomForestClassifier(n_estimators=ne,
                                      max_features=mf,
                                      max_depth=md,
                                      min_samples_split=mss,
                                      min_samples_leaf=msl,
                                      bootstrap=bs,
                                      random_state=self.config['random_state'],
                                      verbose=self.config['verbose'],
                                      n_jobs=self.config['n_jobs'])
    
    def fit(self, 
            X_train, 
            y_train, 
            X_val,
            y_val,
            config=None):
        
        if not config:
            config = self.best_config            
        if self.model == None:
            self.model = self.create(config)
        
        return self.model.fit(X_train, 
                              y_train)
    
    def predict(self,                 
                X_test,
                y_test):
        
        y_pred_prob = self.model.predict_proba(X_test) 
        y_pred = argmax(y_pred_prob , axis = 1)

        self.y_test_true = y_test
        self.y_test_pred = y_pred
        
        if self.le:
            self.y_test_true = self.le.inverse_transform(self.y_test_true)
            self.y_test_pred = self.le.inverse_transform(self.y_test_pred)
        
        self._summary = self.score(y_test, y_pred_prob)
            
        return self._summary, y_pred_prob 