# -*- coding: utf-8 -*-
'''
MAT-classification: Analisys and Classification methods for Multiple Aspect Trajectory Data Mining

The present package offers a tool, to support the user in the task of data analysis of multiple aspect trajectories. It integrates into a unique framework for multiple aspects trajectories and in general for multidimensional sequence data mining methods.
Copyright (C) 2022, MIT license (this portion of code is subject to licensing from source project distribution)

Created on Dec, 2021
Copyright (C) 2022, License GPL Version 3 or superior (see LICENSE file)

@author: Tarlis Portela
'''
# --------------------------------------------------------------------------------
from matclassification.methods.core import *
# --------------------------------------------------------------------------------  

# Hiperparameter Optimization Classifier - For Movelet/Features input data
class MHSClassifier(HSClassifier, MClassifier):
    
    def __init__(self, 
                 name='NN',
                 save_results=False,
                 n_jobs=-1,
                 verbose=False,
                 random_state=42,
                 filterwarnings='ignore'):
        
        super().__init__(name=name, n_jobs=n_jobs, verbose=verbose, random_state=random_state, filterwarnings=filterwarnings, save_results=save_results)
        
        self.save_results = save_results
        
        np.random.seed(seed=random_state)
        random.set_seed(random_state)
