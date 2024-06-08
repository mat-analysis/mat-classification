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
import os 
import numpy as np
import pandas as pd
from numpy import argmax
from datetime import datetime

from tqdm.auto import tqdm
# --------------------------------------------------------------------------------
from tensorflow import random
from matdata.preprocess import trainTestSplit
from matclassification.methods._lib.datahandler import prepareTrajectories
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras.utils import to_categorical

from sklearn.metrics import classification_report
from matclassification.methods._lib.metrics import *
from matclassification.methods._lib.pymove.models import metrics
# --------------------------------------------------------------------------------
import warnings
# --------------------------------------------------------------------------------
from abc import ABC, abstractmethod
# TODO implement rounds

# Simple Abstract Classifier Model
class AbstractClassifier(ABC):
    
    def __init__(self, 
                 name='NN',
                 n_jobs=-1,
                 verbose=0,
                 random_state=42,
                 filterwarnings='ignore'):
        
        self.name = name
        self.y_test_pred = None
        self.model = None
        self.le = None
        
        self.isverbose = verbose >= 0
        
        self.save_results = False # TODO
        self.validate = False
        topK = 5
        
        self.config = dict()
        self.add_config(topK=topK,
                        n_jobs=n_jobs,
                        verbose=verbose,
                        random_state=random_state)
        
        
        if filterwarnings:
            warnings.filterwarnings(filterwarnings)
        
        if self.isverbose:
            print('\n['+self.name+':] Building model')
    
    def add_config(self, **kwargs):
        self.config.update(kwargs)
    
    def duration(self):
        return (datetime.now()-self.start_time).total_seconds() * 1000
    
    def message(self, pbar, text):
        if isinstance(pbar, list):
            print(text)
        else:
            pbar.set_postfix_str(text)

    @abstractmethod
    def create(self):
        
        # **** Method to overrite ****
        print('\n['+self.name+':] Warning! you must overwrite the create() method.')
        self.model = None
        
        return self.model
    
    def clear(self):
        del self.model 

    def fit(self, 
            X_train, 
            y_train, 
            X_val,
            y_val):
        
        self.model = self.create()
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_val)
        
        self.report = self.score(X_val, y_val, y_pred)
        
        return self.report
    
    def predict(self,                 
                X_test,
                y_test):
        
        y_pred = self.model.predict(X_test, y_test)
        
        self.y_test_true = argmax(y_test, axis = 1)
        self.y_test_pred = argmax(y_pred, axis = 1)
        
        if self.le:
            self.y_test_true = self.le.inverse_transform(self.y_test_true)
            self.y_test_pred = self.le.inverse_transform(self.y_test_pred)

        self._summary = self.score(X_test, y_test, y_pred)
            
        return self._summary, y_pred   
    
    def score(self, X_test, y_test, y_pred):
        acc, acc_top5, _f1_macro, _prec_macro, _rec_macro, bal_acc = compute_acc_acc5_f1_prec_rec(y_test, np.array(y_pred), print_metrics=False)
        
        dic_model = {
            'acc': acc,
            'acc_top_K5': acc_top5,
            'balanced_accuracy': bal_acc,
            'precision_macro': _f1_macro,
            'recall_macro': _prec_macro,
            'f1_macro': _rec_macro,
        } 
        
        return pd.DataFrame(dic_model, index=[0])

    @abstractmethod
    def train(self):
        pass

    def test(self,
             rounds=1,
             dir_evaluation='.'):
        
        X_train = self.X_train
        y_train = self.y_train
        
        if self.validate:
            X_val = self.X_val
            y_val = self.y_val
        else:
            X_val = self.X_test
            y_val = self.y_test  
            
        X_test = self.X_test
        y_test = self.y_test
        
        filename = os.path.join(dir_evaluation, 'eval_'+self.name.lower()+'.csv')
        
        if os.path.exists(filename):
            if self.isverbose:
                print('['+self.name+':] Model previoulsy built.')
            # TODO read
            #return self.read_report(filename, prefix='eval_')
        else:
            if self.isverbose:
                print('['+self.name+':] Creating a model to test set')
            
                pbar = tqdm(range(rounds), desc="Model Testing")
            else:
                pbar = list(range(rounds))
                
            random_state = self.config['random_state']
            
            evaluate_report = []
            for e in pbar:
                re = (random_state+e)
                self.config['random_state'] = re
                
                self.message(pbar, 'Round {} of {} (random {})'.format(e, rounds, re))
                
                self.model = self.create()
                
                self.fit(X_train, y_train, X_val, y_val)
                
                eval_report, y_pred = self.predict(X_test, y_test)
                evaluate_report.append(eval_report)
                        
            self.config['random_state'] = random_state
            self.test_report = pd.concat(evaluate_report)
            self.test_report.reset_index(drop=True, inplace=True)
            
            if self.isverbose:
                print('['+self.name+':] Processing time: {} milliseconds. Done.'.format(self.duration()))

            return self.test_report, y_pred

    def summary(self):
        return pd.DataFrame(self.test_report.mean()).T
        
    def save(self, dir_path='.', modelfolder='model'):
        if not os.path.exists(os.path.join(dir_path, modelfolder)):
            os.makedirs(os.path.join(dir_path, modelfolder))
        self.model.save(os.path.join(dir_path, modelfolder, 'model_'+self.name.lower()+'.h5'))

        prediction = self.prediction()
        prediction.to_csv(os.path.join(dir_path, modelfolder, 'model_'+self.name.lower()+'_prediction.csv'), header = True) 
        
        report = self.report()
        classification_report_dict2csv(report, os.path.join(dir_path, modelfolder, 'model_'+self.name.lower()+'_report.csv'), self.approach)
        self.report.to_csv(os.path.join(dir_path, modelfolder, 'model_'+self.name.lower()+'_history.csv'))