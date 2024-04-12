# -*- coding: utf-8 -*-
'''
MAT-analysis: Analisys and Classification methods for Multiple Aspect Trajectory Data Mining

The present package offers a tool, to support the user in the task of data analysis of multiple aspect trajectories. It integrates into a unique framework for multiple aspects trajectories and in general for multidimensional sequence data mining methods.
Copyright (C) 2022, MIT license (this portion of code is subject to licensing from source project distribution)

Created on Dec, 2021
Copyright (C) 2022, License GPL Version 3 or superior (see LICENSE file)

@author: Tarlis Portela
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

from matanalysis.methods._lib.metrics import compute_acc_acc5_f1_prec_rec
from matanalysis.methods._lib.metrics import *

from matanalysis.methods.core import MClassifier

class MRF(MClassifier):
    
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
            
            line = self.score(X_val, y_val, y_pred)
            line['n_tree'] = [n_tree]
            
            lines.append(line)
        
        self.report = pd.concat(lines)
        return self.report
    
#    def predict(self,                 
#                X_test,
#                y_test):
#        
#        y_pred = self.model.predict_proba(X_test)
#        
#        self.y_test_true = y_test #argmax(y_test, axis = 1)
#        self.y_test_pred = argmax(y_pred , axis = 1)
#        
#        if self.le:
#            self.y_test_true = self.le.inverse_transform(self.y_test_true)
#            self.y_test_pred = self.le.inverse_transform(self.y_test_pred)
#            
#        self._summary = self.score(X_test, y_test, self.y_test_pred) #y_pred)
#        
#        return self._summary, y_pred
# --------------------------------------------------------------------------------