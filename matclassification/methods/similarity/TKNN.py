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
from sklearn.neighbors import KNeighborsClassifier
# --------------------------------------------------------------------------------

from matclassification.methods.core import SimilarityClassifier

class TKNN(SimilarityClassifier):
    
    def __init__(self, 
                 k = [1, 3, 5], # Number of neighbors
                 weights = 'distance', # Weight function used in prediction ['uniform', 'distance']
                 
                 save_results=False,
                 n_jobs=1,
                 verbose=0,
                 random_state=42,
                 filterwarnings='ignore'):
        
        super().__init__('TKNN', save_results=save_results, n_jobs=n_jobs, verbose=verbose, random_state=random_state, filterwarnings=filterwarnings)
        
        self.add_config(k=k,
                        weights=weights)

        self.grid_search(k, weights)
        
    def create(self, config):

        k  = config[0]
        w  = config[1]
        
        # Initializing Model
        return KNeighborsClassifier(n_neighbors=k,
                                    weights=w,
                                    metric='precomputed', 
                                    n_jobs=self.config['n_jobs'])