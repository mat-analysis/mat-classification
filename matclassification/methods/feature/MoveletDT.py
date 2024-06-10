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

from matclassification.methods._lib.metrics import compute_acc_acc5_f1_prec_rec
from matclassification.methods._lib.metrics import *

from matclassification.methods.core import MHSClassifier

class MDT(MHSClassifier):
    
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
                      validate=False):
        
        X, y, nattr, num_classes = super().prepare_input(train, test,
                                                         tid_col=tid_col, class_col=class_col, 
#                                                         geo_precision=geo_precision,
                                                         validate=validate)
        
        self.config['features'] = list(filter(lambda c: c not in [tid_col, class_col, 'class'], train.columns))
        
        return X, y, nattr, num_classes
        
    def create(self):
        return DecisionTreeClassifier()
    
    def fit(self, 
            X_train, 
            y_train, 
            X_val,
            y_val):
        
        self.model = self.create()
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_val)
        
        y_val1 = argmax(y_val, axis = 1)
        self.report = self.score(y_val1, y_pred)
        
        return self.report
    
    def training_report(self):
        return None # Disables saving model history file
    
    def plot_tree(self, figsize=(20, 10)):
        import matplotlib.pyplot as plt
        features = self.config['features']
        
        X_train = self.X_train
        y_train = self.le.inverse_transform( argmax(self.y_train, axis = 1) )
        
        model = self.create()
        model.fit(X_train, y_train)

        fig = plt.figure(figsize=figsize)
        tree.plot_tree(model,
                  feature_names=features,
#                  class_names=y_train,#classes,
                  rounded=True, filled=True, proportion=True);
        
        return fig
    
    def graph_tree(self):
        import graphviz
        features = self.config['features']
    
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