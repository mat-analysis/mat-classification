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
from matanalysis.methods._lib.metrics import *
from matanalysis.methods.core import AbstractClassifier, MClassifier, MHPOClassifier, HPOClassifier

from matanalysis.methods import *

class TEC(MHPOClassifier, HPOClassifier):
    
    def __init__(self, 
                 ensembles=['MARC', 'POIS', 'MMLP'],
                 
                 n_jobs=-1,
                 verbose=2,
                 random_state=42,
                 filterwarnings='ignore'):
        
        super().__init__('TEC', n_jobs=n_jobs, verbose=verbose, random_state=random_state, filterwarnings=filterwarnings)
        
        self.add_config(ensembles=ensembles, filterwarnings=filterwarnings)
        
        # tacke future warning, deprecated warning, for sub models
        import warnings
        # tackle some warning
        def warn(*args, **kwargs):
            pass
        warnings.warn = warn
        warnings.simplefilter(action="ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category = DeprecationWarning)

        '''
        TF_CPP_MIN_LOG_LEVEL = 0 to all logs .
        TF_CPP_MIN_LOG_LEVEL = 1 to filter out INFO logs 
        TF_CPP_MIN_LOG_LEVEL = 2 to additionall filter out WARNING 
        TF_CPP_MIN_LOG_LEVEL = 3 to additionally filter out ERROR.
        '''
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        os.environ["PYTHONWARNINGS"] = 'ignore'
        
        
#        import tensorflow as tf
#        tf.keras.backend.clear_session()

    
    def prepare_models(self,
                data,
                tid_col='tid',  # TODO, pass other params
                class_col='label',
                space_geohash=False, # True: Geohash, False: indexgrid
                geo_precision=30,    # Geohash: precision OR IndexGrid: meters
                validate=False):
        
        if validate:
            raise NotImplementedError('['+self.name+':]  with validate=True is not implemented.')
            
        if not self.check_input(data, tid_col, class_col):
            raise ValueError('['+self.name+':] All data input (trajectories and features) must match size and corresponding labels.')
     
        self.models = self.create()
        
        for method, model in self.models.items():
            # MARC and POIS yet have different configurations for input trajectories geo_precision:
            if isinstance(model, POIS) or isinstance(model, MARC):    
                train = data['mat'][0]
                test  = data['mat'][1]
                
                model.prepare_input(train, test, 
                                    tid_col=tid_col, class_col=class_col, 
                                    validate=validate)

            # Movelets as Input:
            elif isinstance(model, MClassifier):
                train = data['movelets'][0]
                test  = data['movelets'][1]
                
                model.prepare_input(train, test,
                                    validate=validate)
                
            # Trajectories as Input:
            elif isinstance(model, HPOClassifier):
                train = data['mat'][0]
                test  = data['mat'][1]
                
                model.prepare_input(train, test,
                                    tid_col=tid_col, class_col=class_col, 
                                    space_geohash=space_geohash, # TODO
                                    validate=validate)
            else:
                print('['+self.name+':] *WARNING* model \'{}\' has no matching data format input.'.format(method))
        
    def create(self):
        module = getattr(__import__('matanalysis'), 'methods')
        
        # create the ensemble models
        ensembles = self.config['ensembles']
        models = dict()
        
        pbar = tqdm(ensembles, desc='['+self.name+':] Model building')
        for method in pbar:
            pbar.set_postfix_str(' - building sub-model {}.'.format(method))
            class_ = getattr(module, method)
            instance = class_(n_jobs=self.config['n_jobs'],
                              random_state=self.config['random_state'],
                              filterwarnings=self.config['filterwarnings'],
                              verbose=0)
#                              verbose=-1)
            models[method] = instance
            
        return models
    
    def sub_fit_predict(self, model, do_fit=True):
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
        
        if do_fit:
#            model.fit(X_train, y_train, X_val, y_val)
            model.train()
            summ, y_pred = model.predict(X_val, y_val)
        else:
            summ, y_pred = model.predict(X_test, y_test)
            
        
        return summ, y_pred
    
    def final_predict(self, y_test, estimators):
        final_pred = estimators[0]
        for i in range(1, len(estimators)):
            final_pred = final_pred + estimators[i]
        
        summ = self.score(None, y_test, np.array(final_pred))
        summ['model'] = self.name
        
        return summ, final_pred
    
    def fit(self):
        
        if not hasattr(self, 'models'):
            raise Exception('['+self.name+':] first imput the data, calling "prepare_models".')
        
        report = pd.DataFrame()
        estimators = []
        
        pbar = tqdm(self.models.items(), desc='['+self.name+':] Model Training')
        for method, model in pbar:
            pbar.set_postfix_str('- training sub-model {}.'.format(method))
            
            summ, y_pred = self.sub_fit_predict(model)
            summ['model'] = method

            estimators.append(y_pred)
            
            report = pd.concat([report, summ])
        
        print('['+self.name+':] \t - Combining models.')
        
        if model.validate:
            y_val = model.y_val
        else:
            y_val = model.y_test
        summ, final_pred = self.final_predict(y_val, estimators)
        self.report = pd.concat([report, summ])
        self.report.reset_index(drop=True, inplace=True)
        
        return self.report
    
    def predict(self):
        
        report = pd.DataFrame()
        estimators = []
        for method, model in self.models.items():
            print('['+self.name+':] \t - Predicting sub-model {}.'.format(method))
            
            summ, y_pred = self.sub_fit_predict(model, do_fit=False)
            summ['model'] = method
            
            estimators.append(y_pred)
            
            report = pd.concat([report, summ])
        
        print('['+self.name+':] \t - Final prediction.')
        
        summ, final_pred = self.final_predict(model.y_test, estimators)
        self._summary = pd.concat([report, summ])
        self._summary.reset_index(drop=True, inplace=True)
        
        return self._summary, final_pred

    def check_input(self, data, tid_col='tid', class_col='label'):
        y_true = {}
        for i, datasets in data.items():
            for j in range(len(datasets)):
                df = datasets[j]
                if i == 'mat':
                    df.sort_values([class_col, tid_col], inplace=True)
                    
                    arr = list(map(lambda df_i: df_i[1][class_col].unique()[0], df.groupby([tid_col])))
                elif i == 'movelets':
                    df.sort_values([class_col, tid_col], inplace=True)
#                    df.drop(columns=[tid_col], inplace=True)
                    
                    arr = df[class_col].values
                elif i == 'pois':
                    raise NotImplementedError('['+self.name+':] check pois data is not implemented.')
                
                if j not in y_true.keys():
                    y_true[j] = []
                y_true[j].append(np.array(arr))
    
        check = True
        for i, y in y_true.items():
            y = np.array(y)
            if len(y) > 1:
                check = check and np.all(np.equal(*y))

        return check , y_true        
# --------------------------------------------------------------------------------