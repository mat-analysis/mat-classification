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
from numpy import argmax
# --------------------------------------------------------------------------------
from sklearn import svm #as SVC 

from matclassification.methods._lib.metrics import *
from matclassification.methods.core import MHSClassifier
# --------------------------------------------------------------------------------

class MSVC(MHSClassifier):
    
    def __init__(self, 
                 kernel="linear", 
                 probability=True,
                 
                 n_jobs=-1,
                 verbose=2,
                 random_state=42,
                 filterwarnings='ignore'):
        
        super().__init__('MSVC', n_jobs=n_jobs, verbose=verbose, random_state=random_state, filterwarnings=filterwarnings)
        
        self.add_config(kernel=kernel,
                        probability=probability)
    
    def create(self):
        
        kernel = self.config['kernel']
        probability = self.config['probability']
        
        model = svm.SVC(kernel=kernel, probability=probability)
        return model
    
    def fit(self, 
            X_train, 
            y_train, 
            X_val,
            y_val):
        
        self.model = self.create()
        
        y_train1 = argmax(y_train, axis = 1)
        
        self.model.fit(X_train, y_train1)
        
        y_pred = self.model.predict_proba(X_val)

        y_val1 = argmax(y_val, axis = 1)
        self.report = self.score(y_val1, y_pred)
        
        return self.report
    
    def predict(self,                 
                X_test,
                y_test):
        
        y_pred = self.model.predict_proba(X_test)
#        y_pred = self.model.predict(X_test)
        
        self.y_test_true = argmax(y_test, axis = 1) #y_test
        self.y_test_pred = argmax(y_pred , axis = 1)
        
        self._summary = self.score(self.y_test_true, y_pred) #self.y_test_pred)
        
        if self.le:
            self.y_test_true = self.le.inverse_transform(self.y_test_true)
            self.y_test_pred = self.le.inverse_transform(self.y_test_pred)
        
        
        return self._summary, y_pred 
    
    def training_report(self):
        return None # Disables saving model history file
# --------------------------------------------------------------------------------