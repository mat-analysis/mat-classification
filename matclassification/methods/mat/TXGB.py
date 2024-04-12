# -*- coding: utf-8 -*-
'''
MAT-analysis: Analisys and Classification methods for Multiple Aspect Trajectory Data Mining

The present package offers a tool, to support the user in the task of data analysis of multiple aspect trajectories. It integrates into a unique framework for multiple aspects trajectories and in general for multidimensional sequence data mining methods.
Copyright (C) 2022, MIT license (this portion of code is subject to licensing from source project distribution)

Created on Dec, 2021
Copyright (C) 2022, License GPL Version 3 or superior (this portion of code is subject to licensing from source project distribution)

@author: Tarlis Portela (adapted)

# Original source:
# Author: Nicksson C. A. de Freitas, 
          Ticiana L. Coelho da Silva, 
          Jose António Fernandes de Macêdo, 
          Leopoldo Melo Junior, 
          Matheus Gomes Cordeiro
# Adapted from: https://github.com/nickssonfreitas/ICAART2021
'''
# --------------------------------------------------------------------------------
import time
import pandas as pd
import numpy as np
from numpy import argmax

from tqdm.auto import tqdm

import itertools
# --------------------------------------------------------------------------------
from matanalysis.methods._lib.pymove.models.classification import XGBoost as xg
# --------------------------------------------------------------------------------

from matanalysis.methods.core import HPOClassifier

class TXGB(HPOClassifier):
    
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
        
        self.grid = list(itertools.product(n_estimators, max_depth, learning_rate, gamma, subsample, 
                                           colsample_bytree, reg_alpha_l1, reg_lambda_l2, eval_metric, esr))
        
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
        return xg.XGBoostClassifier(n_estimators=ne,
                                           max_depth=md,
                                           lr=lr,
                                           gamma=gm,
                                           colsample_bytree=cst,
                                           subsample=ss,
                                           l1=l1,
                                           l2=l2,
                                           random_state=self.config['random_state'],
                                           tree_method=self.config['tree_method'],
                                           eval_metric=loss,
                                           early_stopping_rounds=epch,
                                           num_classes=self.config['num_classes'])
    