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

#import logging
#logging.disable()

from tqdm.auto import tqdm
# --------------------------------------------------------------------------------
from sklearn import preprocessing
# --------------------------------------------------------------------------------
from matclassification.methods._lib.metrics import *
from matclassification.methods.ensemble.TEC import TEC

#from matanalysis.methods import *

class MSTEC(TEC):
    
    def __init__(self, 
                 ensembles=['MARC', 'POIS', 'MMLP'],
                 
                 n_jobs=-1,
                 verbose=2,
                 random_state=42,
                 filterwarnings='ignore'):
        
        super().__init__('MSTEC', ensembles=ensembles, n_jobs=n_jobs, verbose=verbose, random_state=random_state, filterwarnings=filterwarnings)
    
    def fit(self):
        
        #TODO data from T and M
        
        if not hasattr(self, 'models'):
            raise Exception('['+self.name+':] first imput the data, calling "prepare_models".')
        
        report = pd.DataFrame()
        estimators = []
        
        pbar = tqdm(self.models.items(), desc='['+self.name+':] Model Training')
        for method, model in pbar:
            pbar.set_postfix_str('- training sub-model {}.'.format(method))
            
            X_train = model.X_train
            y_train = model.y_train

            if model.validate:
                X_val = model.X_val
                y_val = model.y_val
            else:
                X_val = model.X_test
                y_val = model.y_test  
            
            model.fit(X_train, y_train, X_val, y_val)
            
            summ, y_pred = model.predict(X_val, y_val)

            estimators.append(y_pred)
            
#            summ = model.summary()
            summ['model'] = method
            report = pd.concat([report, summ])
        
        print('['+self.name+':] \t - Combining models.')
        final_pred = estimators[0]
        for i in range(1, len(estimators)):
            final_pred = final_pred + estimators[i]
        y_pred = [np.argmax(f) for f in final_pred]
    
        summ = self.score(None, y_val, np.array(final_pred))
        summ['model'] = self.name
        self.report = pd.concat([report, summ])
        
        return self.report
    
    def predict(self):
        
        report = pd.DataFrame()
        estimators = []
        for method, model in self.models.items():
            print('['+self.name+':] \t - Predicting sub-model {}.'.format(method))
            X_train = model.X_train
            y_train = model.y_train

            if model.validate:
                X_val = model.X_val
                y_val = model.y_val
            else:
                X_val = model.X_test
                y_val = model.y_test 
            
            X_test = model.X_test
            y_test = model.y_test
        
            summ, y_pred = model.predict(X_test, y_test)
            
            estimators.append(y_pred)
            
#            summ = model.summary()
            summ['model'] = method
            report = pd.concat([report, summ])
        
        
        print('['+self.name+':] \t - Final prediction.')
        final_pred = estimators[0]
        for i in range(1, len(estimators)):
            final_pred = final_pred + estimators[i]
        y_pred = [np.argmax(f) for f in final_pred]
        
        self.y_test_true = y_test #argmax(y_test, axis=1)
        self.y_test_pred = y_pred
        
#        return y_test, final_pred
        
        summ = self.score(None, y_test, np.array(final_pred))
        summ['model'] = self.name
        self._summary = pd.concat([report, summ])
        
        return self._summary, final_pred
    
#    def score(self, X_test, y_test, y_pred):
#        acc = accuracy_score(y_test, y_pred, normalize=True)
#        acc_top5 = -1 #accuracy_top_k(y_test, y_pred, K=5) #calculateAccTop5(self.model, X_test, y_test, self.config['topK'])
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
# --------------------------------------------------------------------------------