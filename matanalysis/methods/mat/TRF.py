# -*- coding: utf-8 -*-
'''
MAT-analysis: Analisys and Classification methods for Multiple Aspect Trajectory Data Mining

The present package offers a tool, to support the user in the task of data analysis of multiple aspect trajectories. It integrates into a unique framework for multiple aspects trajectories and in general for multidimensional sequence data mining methods.
Copyright (C) 2022, MIT license (this portion of code is subject to licensing from source project distribution)

Created on Dec, 2021
Copyright (C) 2022, License GPL Version 3 or superior (this portion of code is subject to licensing from source project distribution)

@author: Tarlis Portela (adapted)

# Original source:
# Author: Nicksson C. A. de Freitas, 
          Ticiana L. Coelho da Silva, 
          Jose António Fernandes de Macêdo, 
          Leopoldo Melo Junior, 
          Matheus Gomes Cordeiro
# Adapted from: https://github.com/nickssonfreitas/ICAART2021
'''
# --------------------------------------------------------------------------------
import time
import pandas as pd
import numpy as np
from numpy import argmax

from tqdm.auto import tqdm

import itertools
# --------------------------------------------------------------------------------
from matanalysis.methods._lib.pymove.models.classification import RandomForest as rf
# --------------------------------------------------------------------------------

from matanalysis.methods.core import HPOClassifier

class TRF(HPOClassifier):
    
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

        self.grid = list(itertools.product(n_estimators, max_depth, min_samples_split, 
                                           min_samples_leaf, max_features, bootstrap))
        
        self.model = None
        
    def create(self, config):

        ne  = config[0]
        md  = config[1]
        mss = config[2]
        msl = config[3]
        mf  = config[4]
        bs  = config[5]
        
        #Initializing Neural Network
        return rf.RFClassifier(n_estimators=ne,
                             max_depth=md,
                             max_features=mf,
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
        if not self.model:
            self.model = self.create(config)
        
        return self.model.fit(X_train, 
                              y_train)
    
    def test(self,
             rounds=10,
             dir_evaluation='.'):
        
        # Saving Temporarely:
        X_train = self.X_train
        y_train = self.y_train
        X_val = self.X_val
        y_val = self.y_val
        
        if self.validate:
            self.X_train = np.concatenate([X_train, X_val])
            self.y_train = np.concatenate([y_train, y_val])
           
        # Do testing:
        super().test(rounds, dir_evaluation)
        
        # Undo:
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        
        return self.report