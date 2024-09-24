# -*- coding: utf-8 -*-
'''
MAT-analysis: Analisys and Classification methods for Multiple Aspect Trajectory Data Mining

The present package offers a tool, to support the user in the task of data analysis of multiple aspect trajectories. It integrates into a unique framework for multiple aspects trajectories and in general for multidimensional sequence data mining methods.
Copyright (C) 2022, MIT license (this portion of code is subject to licensing from source project distribution)

Created on Dec, 2021
Copyright (C) 2022, License GPL Version 3 or superior (this portion of code is subject to licensing from source project distribution)

@author: Tarlis Portela
'''
# --------------------------------------------------------------------------------
import time
import pandas as pd
import numpy as np
from numpy import argmax

from tqdm.auto import tqdm

import itertools
# --------------------------------------------------------------------------------
from sklearn.svm import SVC
# --------------------------------------------------------------------------------

from matclassification.methods.core import SimilarityClassifier

class TSVC(SimilarityClassifier):
    
    def __init__(self, 
                 # TODO: Params here
                 
                 save_results=False,
                 n_jobs=1,
                 verbose=0,
                 random_state=42,
                 filterwarnings='ignore'):
        
        super().__init__('TSVC', save_results=save_results, n_jobs=n_jobs, verbose=verbose, random_state=random_state, filterwarnings=filterwarnings)
        
#        self.add_config(k=k)

#        self.grid_search(k, weights)
        
    def create(self, config):

#        k  = config[0]
#        w  = config[1]
        
        # Initializing Model
        return SVC(probability=True)
    
#    def fit(self, 
#            X_train, 
#            y_train, 
#            X_val=None,
#            y_val=None,
#            config=None):
#        
#        if not config:
#            config = self.best_config            
#        if self.model == None:
#            self.model = self.create(config)
#        
#        return self.model.fit(X_train, y_train)
#    
#    def predict(self,                 
#                X_test,
#                y_test):
#        
#        y_pred_prob = self.model.predict_proba(X_test) 
#        y_pred = argmax(y_pred_prob , axis = 1)
#
#        self.y_test_true = y_test
#        self.y_test_pred = y_pred
#        
#        if self.le:
#            self.y_test_true = self.le.inverse_transform(self.y_test_true)
#            self.y_test_pred = self.le.inverse_transform(self.y_test_pred)
#        
#        self._summary = self.score(y_test, y_pred_prob)
#            
#        return self._summary, y_pred_prob 