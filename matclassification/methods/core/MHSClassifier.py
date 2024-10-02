# -*- coding: utf-8 -*-
'''
MAT-Tools: Python Framework for Multiple Aspect Trajectory Data Mining

The present package offers a tool, to support the user in the task of data analysis of multiple aspect trajectories. It integrates into a unique framework for multiple aspects trajectories and in general for multidimensional sequence data mining methods.
Copyright (C) 2022, MIT license (this portion of code is subject to licensing from source project distribution)

Created on Dec, 2021
Copyright (C) 2022, License GPL Version 3 or superior (see LICENSE file)

Authors:
    - Tarlis Portela
'''
# --------------------------------------------------------------------------------
from matclassification.methods.core import *
# --------------------------------------------------------------------------------  

# Hiperparameter Optimization Classifier - For Movelet/Features input data
class MHSClassifier(HSClassifier, MClassifier):
    """
    Movelet Hyperparameter Optimization Classifier.

    The MHSClassifier is a hybrid classifier that integrates hyperparameter optimization (inherited from HSClassifier) 
    with data preprocessing and feature preparation (inherited from MClassifier) for movelet input. 
    It is specifically designed to handle movelet-based or feature-based trajectory data input and supports 
    hyperparameter tuning across different configurations.

    Attributes:
    -----------
    save_results : bool
        If True, saves the training and evaluation results (default: False).
    
    Parameters:
    -----------
    name : str, optional
        Name of the classifier model (default: 'NN').
    save_results : bool, optional
        Flag to indicate whether results should be saved (default: False).
    n_jobs : int, optional
        Number of parallel jobs to run (default: -1 for using all processors).
    verbose : bool, optional
        Flag to control verbosity of the model output (default: False).
    random_state : int, optional
        Random seed for reproducibility (default: 42).
    filterwarnings : str, optional
        Warning filter level (default: 'ignore').

    Methods:
    --------
    Inherits methods from both HSClassifier and MClassifier for training, testing, and handling data.
    """
    
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
