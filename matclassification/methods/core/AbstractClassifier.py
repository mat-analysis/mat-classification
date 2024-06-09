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
#from matclassification.methods._lib.pymove.models import metrics
# --------------------------------------------------------------------------------
import warnings
# --------------------------------------------------------------------------------
from abc import ABC, abstractmethod
# TODO implement rounds

from enum import Enum, auto
class Approach(Enum):
    NN   = auto()
    RF   = auto()
    DT   = auto()
    SVC  = auto()

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
        
        # When creating new methods, choose an approach category
        self.approach = Approach.NN
        
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
    
    @property
    def labels(self):
        return set(self.y_test_true)

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
        
        self.report = self.score(y_val, y_pred)
        
        return self.report
    
    def predict(self,                 
                X_test,
                y_test):
        
#        y_pred = self.model.predict_proba(X_test)
        y_pred = self.model.predict(X_test)
    
        if y_test.ndim == 2:
            y_test = argmax(y_test, axis = 1)
    
        self._summary = self.score(y_test, y_pred) #self.y_test_pred)
        
        self.y_test_true = y_test
        self.y_test_pred = argmax(y_pred , axis = 1)
        
        if self.le:
            self.y_test_true = self.le.inverse_transform(self.y_test_true)
            self.y_test_pred = self.le.inverse_transform(self.y_test_pred)
        
        return self._summary, y_pred 

    
    def score(self, y_test, y_pred):
        acc, acc_top5, _f1_macro, _prec_macro, _rec_macro, bal_acc = compute_acc_acc5_f1_prec_rec(np.array(y_test), np.array(y_pred), print_metrics=False)
        
        dic_model = {
            'acc': acc,
            'acc_top_K5': acc_top5,
            'balanced_accuracy': bal_acc,
            'precision_macro': _f1_macro,
            'recall_macro': _prec_macro,
            'f1_macro': _rec_macro,
        } 
        
        return pd.DataFrame(dic_model, index=[0])

    def summary(self):
        return pd.DataFrame(self.test_report.mean()).T
    
    def cm(self): # Confusion Matrix Plot
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
        
        cfm = ConfusionMatrixDisplay(
            confusion_matrix = confusion_matrix(self.y_test_true, self.y_test_pred), 
            display_labels = self.labels
        )
        return cfm.plot()
        
    def save_model(self, dir_path='.', modelfolder='model', model_name=''):
        if not os.path.exists(os.path.join(dir_path, modelfolder)):
            os.makedirs(os.path.join(dir_path, modelfolder))
        
        self.model.save(os.path.join(dir_path, modelfolder, 'model_'+self.name.lower()+'_'+str(model_name)+'.h5'))
        
    def save(self, dir_path='.', modelfolder='model'):
        if not os.path.exists(os.path.join(dir_path, modelfolder)):
            os.makedirs(os.path.join(dir_path, modelfolder))

        prediction = self.prediction_report() 
        if prediction is not None:
            prediction.to_csv(os.path.join(dir_path, modelfolder, 'model_'+self.name.lower()+'_prediction.csv')) 
        
        report = self.classification_report()
        if report is not None:
            report.to_csv(os.path.join(dir_path, modelfolder, 'model_'+self.name.lower()+'_report.csv'), index = False)
            
        train_report = self.training_report()
        if train_report is not None:
            train_report.to_csv(os.path.join(dir_path, modelfolder, 'model_'+self.name.lower()+'_history.csv'))
            
        test_report = self.testing_report()
        if test_report is not None:
            test_report.to_csv(os.path.join(dir_path, modelfolder, 'model_'+self.name.lower()+'_summary.csv'))
    
    def prediction_report(self):
        df = pd.DataFrame(self.y_test_true, self.y_test_pred, columns=['true_label'])
        df.rename_axis(index='prediction', inplace=True)
        return df
    def classification_report(self):
        report = classification_report(self.y_test_true, self.y_test_pred, output_dict=True, zero_division=False)
        return classification_report_dict2df(report, self.approach.name.upper())
    def training_report(self):
        return self.report
    def testing_report(self):
        return self.test_report