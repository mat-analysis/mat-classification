# -*- coding: utf-8 -*-
'''
MAT-Tools: Python Framework for Multiple Aspect Trajectory Data Mining

The present package offers a tool, to support the user in the task of data analysis of multiple aspect trajectories. It integrates into a unique framework for multiple aspects trajectories and in general for multidimensional sequence data mining methods.
Copyright (C) 2022, MIT license (this portion of code is subject to licensing from source project distribution)

Created on Dec, 2021
Copyright (C) 2022, License GPL Version 3 or superior (see LICENSE file)

Authors:
    - Tarlis Portela
    - Carlos Andrés Ferrero (adapted)
'''
# --------------------------------------------------------------------------------
import time
import pandas as pd
import numpy as np
from numpy import argmax

from tqdm.auto import tqdm
# --------------------------------------------------------------------------------
from sklearn import preprocessing
# --------------------------------------------------------------------------------
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

from matclassification.methods._lib.metrics import compute_acc_acc5_f1_prec_rec
from matclassification.methods._lib.metrics import *

from matclassification.methods.core import MHSClassifier

class MRFHP(MHSClassifier):
    """
    Movelet Random Forest with Hyperparameter Optimization (MRFHP) Classifier, extending MHSClassifier, designed to
    optimize hyperparameters for movelet-based classification using RandomizedSearchCV.

    Parameters:
    -----------
    n_estimators : list of int, optional (default=[300, 350, 400, 450, 500, 550, 600])
        A list of the number of trees (estimators) to be used in the Random Forest.

    max_features : list of str, optional (default=['auto', 'sqrt'])
        Number of features to consider at every split. Options include 'auto', 'sqrt', etc.

    max_depth : list of int, optional (default=[30])
        Maximum number of levels in each decision tree. The default is 30, and None is added to allow full growth.

    min_samples_split : list of int, optional (default=[2, 4, 6])
        The minimum number of samples required to split a node.

    min_samples_leaf : list of int, optional (default=[2, 3, 4])
        The minimum number of samples required to be at a leaf node.

    bootstrap : list of bool, optional (default=[True, False])
        Whether to bootstrap samples when building trees.

    criterion : list of str, optional (default=['entropy', 'gini'])
        The function to measure the quality of a split. Options are 'gini' or 'entropy'.

    n_jobs : int, optional (default=-1)
        The number of parallel jobs to run for computation. -1 means using all processors.

    verbose : int, optional (default=2)
        Verbosity level for logging and output during model training and hyperparameter tuning.

    random_state : int, optional (default=42)
        Seed used by the random number generator to ensure reproducibility.

    filterwarnings : str, optional (default='ignore')
        Controls the filter for output warnings.

    Methods:
    --------
    create():
        Creates a Random Forest model wrapped with RandomizedSearchCV to search for optimal hyperparameters.

    fit(X_train, y_train, X_val, y_val):
        Fits the model using the training data and performs hyperparameter optimization using cross-validation. Returns a report on the search results and sets the model to the best estimator.

    """
    
    def __init__(self, 
                 # Number of trees in random forest
                 n_estimators = [300,350,400,450,500,550,600],
                 # Number of features to consider at every split
                 max_features = ['auto', 'sqrt'],
                 # Maximum number of levels in tree
                 max_depth = [30],
                 # Minimum number of samples required to split a node
                 min_samples_split = [2,4,6],
                 # Minimum number of samples required at each leaf node
                 min_samples_leaf = [2,3,4],
                 # Method of selecting samples for training each tree
                 bootstrap = [True, False],
                 # Create the random grid
                 criterion = ['entropy','gini'],
                 
                 n_jobs=-1,
                 verbose=2,
                 random_state=42,
                 filterwarnings='ignore'):
        super().__init__('MRFHP', n_jobs=n_jobs, verbose=verbose, random_state=random_state, filterwarnings=filterwarnings)
        
        max_depth.append(None)
        
        self.add_config(n_estimators=n_estimators,
                        max_features=max_features,
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        min_samples_leaf=min_samples_leaf,
                        bootstrap=bootstrap,
                        criterion=criterion)
        
        self.random_grid = {'n_estimators': n_estimators,
                            'max_features': max_features,
                            'max_depth': max_depth,
                            'min_samples_split': min_samples_split,
                            'min_samples_leaf': min_samples_leaf,
                            'bootstrap': bootstrap,
                            'criterion': criterion}
        
    def create(self):
        
        rf = RandomForestClassifier(verbose=self.config['verbose'], random_state = 1)
        model = RandomizedSearchCV(estimator = rf, param_distributions = self.random_grid, n_iter = 50, cv = 3, verbose=self.config['verbose'], random_state=1, n_jobs = -1)
        
        return model
        
    def fit(self, 
            X_train, 
            y_train, 
            X_val,
            y_val):

        self.model = self.create()
        self.model.fit(X_train, y_train)
        
        self.report = pd.DataFrame(self.model.cv_results_)
    
        if self.isverbose:
            print(self.model.best_params_)

        self.model = self.model.best_estimator_
        
        return self.report
# --------------------------------------------------------------------------------