# -*- coding: utf-8 -*-
'''
MAT-Tools: Python Framework for Multiple Aspect Trajectory Data Mining

The present package offers a tool, to support the user in the task of data analysis of multiple aspect trajectories. It integrates into a unique framework for multiple aspects trajectories and in general for multidimensional sequence data mining methods.
Copyright (C) 2022, MIT license (this portion of code is subject to licensing from source project distribution)

Created on Dec, 2021
Copyright (C) 2022, License GPL Version 3 or superior (this portion of code is subject to licensing from source project distribution)

Authors:
    - Tarlis Portela
    - Original source:
        - Nicksson C. A. de Freitas, 
        - Ticiana L. Coelho da Silva, 
        - Jose António Fernandes de Macêdo, 
        - Leopoldo Melo Junior, 
        - Matheus Gomes Cordeiro
    - Adapted from: https://github.com/nickssonfreitas/ICAART2021
'''
# --------------------------------------------------------------------------------
import time
import pandas as pd
import numpy as np
from numpy import argmax

from tqdm.auto import tqdm

import itertools
# --------------------------------------------------------------------------------
import xgboost as xgb
import subprocess
# --------------------------------------------------------------------------------
from matclassification.methods.core import THSClassifier

class TXGB(THSClassifier):
    """
    TXGB: Trajectory XGBoost Classifier

    The `TXGB` class is an implementation of the XGBoost classifier, tailored specifically for trajectory classification tasks. It utilizes the efficient gradient boosting algorithm provided by XGBoost and supports a wide range of tunable hyperparameters. 
    
    The model selection process is driven by grid search, making it adaptable to various data configurations and problem complexities.

    Parameters
    ----------
    n_estimators : list, default=[2000]
        Number of boosting rounds.
    max_depth : list, default=[3, 5]
        Maximum depth of a tree.
    learning_rate : list, default=[0.01]
        Step size shrinkage used in update to prevent overfitting.
    gamma : list, default=[0.0, 1, 5]
        Minimum loss reduction required to make a further partition on a leaf node.
    subsample : list, default=[0.1, 0.2, 0.5, 0.8]
        Subsample ratio of the training instance.
    colsample_bytree : list, default=[0.5, 0.7]
        Subsample ratio of columns when constructing each tree.
    reg_alpha_l1 : list, default=[1.0]
        L1 regularization term on weights.
    reg_lambda_l2 : list, default=[100]
        L2 regularization term on weights.
    eval_metric : list, default=['merror', 'mlogloss']
        Evaluation metrics used to monitor performance (merror: classification error, mlogloss: log loss).
    tree_method : str, default='auto'
        The tree construction algorithm used by XGBoost (e.g., 'auto', 'gpu_hist').
    esr : list, default=[20]
        Early stopping rounds (used to stop training early if no improvement is seen).
    save_results : bool, default=False
        Whether to save the results of the classifier.
    n_jobs : int, default=-1
        Number of parallel threads used by XGBoost.
    verbose : int, default=0
        Verbosity of XGBoost training output.
    random_state : int, default=42
        Controls the randomness of the model.
    filterwarnings : str, default='ignore'
        Whether to suppress or display warnings during model training and evaluation.

    Methods
    -------
    create(config):
        Initializes the XGBoost classifier with the given configuration.
    fit(X_train, y_train, X_val, y_val, config=None):
        Trains the XGBoost classifier on the provided training data, with optional early stopping.
    """
    
    def __init__(self, 
                 n_estimators = [2000], 
                 max_depth = [3, 5], 
                 learning_rate = [0.01], 
                 gamma = [0.0, 1, 5], 
                 subsample = [0.1, 0.2, 0.5, 0.8], 
                 colsample_bytree = [0.5 , 0.7], 
                 reg_alpha_l1 = [1.0], #[0.0, 0.01, 1.0], 
                 reg_lambda_l2 = [100], #[0.0, 1.0, 100], 
                 eval_metric = ['merror', 'mlogloss'], #merror #(wrong cases)/#(all cases) Multiclass classification error // mlogloss:
                 tree_method = 'auto', 
                 esr = [20],
                 save_results=False,
                 n_jobs=-1,
                 verbose=0,
                 random_state=42,
                 filterwarnings='ignore'):
        
        super().__init__('TXGB', save_results=save_results, n_jobs=n_jobs, verbose=verbose, random_state=random_state, filterwarnings=filterwarnings)
        
        if tree_method == 'auto':
            tree_method = 'hist' if nvidia_gpu_count() == 0 else 'gpu_hist'
        else:
            tree_method = 'gpu_hist'
        
        self.add_config(n_estimators=n_estimators, 
                        max_depth=max_depth, 
                        learning_rate=learning_rate, 
                        gamma=gamma, 
                        subsample=subsample, 
                        colsample_bytree=colsample_bytree, 
                        reg_alpha_l1=reg_alpha_l1, 
                        reg_lambda_l2=reg_lambda_l2, 
                        eval_metric=eval_metric, 
                        tree_method=tree_method, 
                        esr=esr)
        
        self.grid_search(n_estimators, max_depth, learning_rate, gamma, subsample, 
                         colsample_bytree, reg_alpha_l1, reg_lambda_l2, eval_metric, esr)
#        self.grid = list(itertools.product(n_estimators, max_depth, learning_rate, gamma, subsample, 
#                                           colsample_bytree, reg_alpha_l1, reg_lambda_l2, eval_metric, esr))
        
        self.model = None
        
    def create(self, config):

        ne=config[0]
        md=config[1]
        lr=config[2]
        gm=config[3]
        ss=config[4]
        cst=config[5]
        l1=config[6]
        l2=config[7]
        loss=config[8]
        epch=config[9] 
        
        #Initializing Neural Network
        model = xgb.XGBClassifier(max_depth=md, 
                                  learning_rate=lr,
                                  n_estimators=ne, 
                                  tree_method=self.config['tree_method'],
                                  subsample=ss, 
                                  gamma=gm,
                                  reg_alpha_l1=l1, 
                                  reg_lambda_l2=l2,
                                  n_jobs=-1, 
                                  early_stopping_rounds=epch,
                                  random_state=self.config['random_state'],
                                  eval_metric=loss,
                                  objective='multi:softmax',
                                  num_class=self.config['num_classes']) # Added by Tarlis
        
        return model
    
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
            
        eval_set = [(X_train, y_train), (X_val, y_val)]
        return self.model.fit(X_train, y_train, 
                              eval_set=eval_set, 
                              verbose=self.config['verbose'])


def nvidia_gpu_count():
    try:
        return str(subprocess.check_output(["nvidia-smi", "-L"])).count('UUID')
    except:
        return 0