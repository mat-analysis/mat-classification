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

from matclassification.methods._lib.metrics import compute_acc_acc5_f1_prec_rec
from matclassification.methods._lib.metrics import *

from matclassification.methods.core import MHSClassifier

class MRF(MHSClassifier):
    """
    Movelet Random Forest (MRF) Classifier, extending MHSClassifier, designed for movelet-based classification using 
    a Random Forest model with multiple configurations for the number of trees.

    Parameters:
    -----------
    n_estimators : list of int, optional (default=[300])
        A list specifying the different number of trees (estimators) to be used in the Random Forest.

    n_jobs : int, optional (default=-1)
        The number of parallel jobs to run for computation. -1 means using all processors.

    verbose : int, optional (default=0)
        Verbosity level for logging and output during the model's training process.

    random_state : int, optional (default=42)
        Random seed for reproducibility of results.

    filterwarnings : str, optional (default='ignore')
        Controls the filter for output warnings.

    Methods:
    --------
    create(n_tree=None):
        Creates and returns a Random Forest model with a specified number of trees (n_tree). If not provided, defaults to the configuration's 'n_tree'.

    fit(X_train, y_train, X_val, y_val):
        Fits the Random Forest model on the training data, evaluates it on the validation data, and logs the performance for each configuration of n_estimators.
        Returns a report on the evaluation metrics.

    """
    
    def __init__(self, 
                 n_estimators = [300],
                 
                 n_jobs=-1,
                 verbose=0,
                 random_state=42,
                 filterwarnings='ignore'):
        
        super().__init__('MRF', n_jobs=n_jobs, verbose=verbose, random_state=random_state, filterwarnings=filterwarnings)
        
        
        
        self.add_config(n_estimators=n_estimators,
                       n_tree=n_estimators[0]) # TODO temporary fix, needs to be trained for the best config
        
    def create(self, n_tree=None):
        
        if not n_tree:
            n_tree = self.config['n_tree']
        
        verbose = self.config['verbose']
        random_state = self.config['random_state']
        
        return RandomForestClassifier(verbose=verbose, n_estimators = n_tree, n_jobs = -1, random_state = 1, criterion = 'gini', bootstrap=True)

        
    def fit(self, 
            X_train, 
            y_train, 
            X_val,
            y_val):

        lines = list()
        
        n_estimators = self.config['n_estimators']
        
        n_jobs = self.config['n_jobs']
    
        line = []
        for n_tree in n_estimators:
            self.model = self.create(n_tree)
            self.model.fit(X_train, y_train)
            
            y_pred = self.model.predict(X_val)
            
            line = self.score(y_val.argmax(axis = 1), y_pred)
            line['n_tree'] = [n_tree]
            
            lines.append(line)
        
        self.report = pd.concat(lines)
        return self.report
    
    def training_report(self):
        return None # Disables saving model history file
    
# --------------------------------------------------------------------------------