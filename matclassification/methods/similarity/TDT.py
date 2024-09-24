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
import graphviz
from sklearn.tree import DecisionTreeClassifier, export_graphviz
# --------------------------------------------------------------------------------

from matclassification.methods.core import SimilarityClassifier

class TDT(SimilarityClassifier):
    
    def __init__(self, 
                 # TODO: Params here
                 
                 save_results=False,
                 n_jobs=1,
                 verbose=0,
                 random_state=42,
                 filterwarnings='ignore'):
        
        super().__init__('TDT', save_results=save_results, n_jobs=n_jobs, verbose=verbose, random_state=random_state, filterwarnings=filterwarnings)
        
#        self.add_config(k=k)

#        self.grid_search(k, weights)
        
    def create(self, config):

#        k  = config[0]
#        w  = config[1]
        
        # Initializing Model
        return DecisionTreeClassifier()
    
    def tree(self): 
#        X_train = self.X_train
        #y_train = self.le.inverse_transform( argmax(self.y_train, axis = 1) )
        
        features = list(map(lambda i: 't'+str(i), self.y_train))
        
        if self.model == None:
            model = self.create()
            model.fit(self.X_train, y_train)
        else:
            model = self.model
        
        # DOT data
        dot_data = export_graphviz(model, out_file=None, 
                                   feature_names=features,  
#                                  class_names=classes,
                                   filled=True)

        # Draw graph
        graph = graphviz.Source(dot_data, format="png") 
        return graph