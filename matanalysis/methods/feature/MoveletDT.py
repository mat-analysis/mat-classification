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
from sklearn import tree
## This class have inner imports
#from tensorflow.keras.utils import to_categorical
# --------------------------------------------------------------------------------
from sklearn.tree import DecisionTreeClassifier

from matanalysis.methods._lib.metrics import compute_acc_acc5_f1_prec_rec
from matanalysis.methods._lib.metrics import *

from matanalysis.methods.core import MClassifier

class MDT(MClassifier):
    
    def __init__(self, 
                 n_jobs=-1,
                 verbose=2,
                 random_state=42,
                 filterwarnings='ignore'):
        super().__init__('MDT', n_jobs=n_jobs, verbose=verbose, random_state=random_state, filterwarnings=filterwarnings)
    
    def prepare_input(self,
                      train, test,
                      
                      tid_col='tid', class_col='label',
#                      space_geohash=False, # For future implementation
                      geo_precision=30,
                      validate=True):
        
        X, y, nattr, num_classes = super().prepare_input(train, test,
                                                         tid_col=tid_col, class_col=class_col, 
                                                         geo_precision=geo_precision,
                                                         validate=validate)
        
        self.config['features'] = list(filter(lambda c: c not in [tid_col, class_col, 'class'], train.columns))
        
        return X, y, nattr, num_classes
        
    def create(self):
        return DecisionTreeClassifier()
    
#    def score(self, X_test, y_test, y_pred):
#        acc, acc_top5, _f1_macro, _prec_macro, _rec_macro, bal_acc = compute_acc_acc5_f1_prec_rec(y_test, np.array(y_pred))
#        
#        dic_model = {
#            'acc': acc,
#            'acc_top_K5': acc_top5,
#            'balanced_accuracy': bal_acc,
#            'precision_macro': _f1_macro,
#            'recall_macro': _prec_macro,
#            'f1_macro': _rec_macro,
#        } 
#        
#        return pd.DataFrame(dic_model, index=[0])
    
#    def score(self, X_test, y_test, y_pred):
#        acc = self.model.score(X_test,y_test)
#        acc_top5 = calculateAccTop5(self.model, X_test, y_test, self.config['topK'])
#        bal_acc = balanced_accuracy(y_test, y_pred)
#        _f1_macro = f1_score(y_test, y_pred, average='macro')
#        _prec_macro = precision_score(y_test, y_pred, average='macro', zero_division=1)
#        _rec_macro = recall_score(y_test, y_pred, average='macro')
#        
#        dic_model = {
#            'acc': acc,
#            'acc_top_K5': acc_top5,
#            'balanced_accuracy': bal_acc,
#            'precision_macro': _f1_macro,
#            'recall_macro': _prec_macro,
#            'f1_macro': _rec_macro,
#        } 
#        
#        return pd.DataFrame(dic_model, index=[0])
    
    def plot_tree(self, figsize=(20, 10)):
        import matplotlib.pyplot as plt
        features = self.config['features']
        
#        classes = argmax(self.y_train, axis = 1)
#        if self.le:
#            classes = self.le.inverse_transform(classes)
##        classes  = list(map(lambda l: str(l), set(self.y_train)))
        
        X_train = self.X_train
        y_train = self.le.inverse_transform( argmax(self.y_train, axis = 1) )
        
        model = self.create()
        model.fit(X_train, y_train)
        
#        model.predict(X_val)

        fig = plt.figure(figsize=figsize)
        tree.plot_tree(model,
                  feature_names=features,
#                  class_names=y_train,#classes,
                  rounded=True, filled=True, proportion=True);
        
        return fig
    
    def graph_tree(self):
        import graphviz
        features = self.config['features']
        
#        classes = argmax(self.y_train, axis = 1)
#        if self.le:
#            classes = self.le.inverse_transform(classes)
##        classes  = list(map(lambda l: str(l), set(self.y_train)))

        X_train = self.X_train
        y_train = self.le.inverse_transform( argmax(self.y_train, axis = 1) )
        
        model = self.create()
        model.fit(X_train, y_train)
        
        # DOT data
        dot_data = tree.export_graphviz(model, out_file=None, 
                                        feature_names=features,  
#                                        class_names=classes,
                                        filled=True)

        # Draw graph
        graph = graphviz.Source(dot_data, format="png") 
        return graph
    
#    def fit(self, 
#            X_train, 
#            y_train, 
#            X_val,
#            y_val):
#        
#        self.model = self.create()
#        self.model.fit(X_train, y_train)
#        
#        y_pred = self.model.predict(X_val)
#        
#        self.report = self.score(X_val, y_val, y_pred)
#        
#        return self.report
    
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
#        self.tree = self.model.tree_
#        
#        return self._summary, y_pred
    
# For Future implementation
#    def dtreeviz_tree(self):
#        import dtreeviz
#        
#        features = self.config['features']
#        classes  = list(map(lambda l: str(l), set(self.y_train)))
#        
#        y_test = list(map(lambda x: self.y_test.index(x), self.y_test))
#
#        viz = dtreeviz.model(self.model, self.X_test, y_test,
#                        target_name="target",
#                        feature_names=features,
#                        class_names=classes)
#
#        return viz
# --------------------------------------------------------------------------------