# -*- coding: utf-8 -*-
'''
MAT-analysis: Analisys and Classification methods for Multiple Aspect Trajectory Data Mining

The present package offers a tool, to support the user in the task of data analysis of multiple aspect trajectories. It integrates into a unique framework for multiple aspects trajectories and in general for multidimensional sequence data mining methods.
Copyright (C) 2022, MIT license (this portion of code is subject to licensing from source project distribution)

Created on Dec, 2021
Copyright (C) 2022, License GPL Version 3 or superior (see LICENSE file)

@author: Tarlis Portela
@author: Carlos Andres Ferreira (adapted)
'''
# --------------------------------------------------------------------------------
import os 
import numpy as np
import pandas as pd
from numpy import argmax
# --------------------------------------------------------------------------------
from sklearn.metrics import classification_report

from matanalysis.methods._lib.pymove.models import metrics
from matanalysis.methods._lib.metrics import classification_report_csv, classification_report_dict2csv
# --------------------------------------------------------------------------------
class ModelClassifier(object):
    
    def __init__(self, 
                 name='NN',
                 n_jobs=-1,
                 verbose=False,
                 random_state=42):
        
        self.name = name
        
        self.config = dict()
        self.add_config(n_jobs,verbose,random_state)
        
        if verbose:
            print('\n['+self.name+':] Building model')
        self.start_time = time.time()
        
        #assert (eval_metric == 'merror') | (eval_metric == 'mlogloss'), "ERR: invalid loss, set loss as mlogloss or merror" 

        #print('[MODEL:] Starting model training, {} iterations'.format(total))

        
    
    def add_config(**params):
        self.config.update(params)
    
    def duration(self):
        return (datetime.now()-self.start_time).total_seconds() * 1000
        
    def fit(self, 
            X_train, 
            y_train, 
            X_val,
            y_val, 
            verbose=True):
        
        eval_set = [(X_train, y_train), (X_val, y_val)]
        
        self.model.fit(X_train, y_train, 
                      eval_set=eval_set,
                      verbose=verbose) 
        if verbose:
            print('['+self.name+':] Processing time: {} milliseconds. Done.'.format(self.duration()))
        
    def predict(self,                 
                X_test,
                y_test):
        
        y_pred = self.model.predict(X_test) 
    
        classification_report = metrics.compute_acc_acc5_f1_prec_rec(y_test, y_pred)
        return classification_report, y_pred
    
#    def container(self):
#        return ModelContainer(self.classifier, self.y_test_true, self.x_test, cls_history=history.history, approach='MLP', le=le)

# DEPRECATED:
class ModelContainer:
    def __init__(self, classifier, y_test_true, x_test, cls_history=None, approach='NN', le=None):
        self.classifier = classifier
        self.x_test = x_test
        self.y_test_true = y_test_true
        self.y_test_pred = None
        self.le = le
        
        self.approach = approach
        
        if not isinstance(cls_history, pd.DataFrame):
            cls_history = pd.DataFrame(cls_history)
        self.cls_history = cls_history
        
        self.predicted = False
        
    def predict(self):
        self.y_test_true = argmax(self.y_test_true, axis = 1)
        self.y_test_pred = argmax( self.classifier.predict(self.x_test) , axis = 1)
        
    def label_decode(self):
        if self.le:
            self.y_test_true = self.le.inverse_transform(self.y_test_true)
            self.y_test_pred = self.le.inverse_transform(self.y_test_pred)
        
    def prediction(self):
        if not self.predicted:
            self.predict()
            self.label_decode()
                
        self.predicted = True
        return pd.DataFrame(self.y_test_true,self.y_test_pred)
    
    def report(self):
        #classification_report = metrics.compute_acc_acc5_f1_prec_rec(self.y_test_true, np.array(self.y_test_pred))
        return classification_report(self.y_test_true, self.y_test_pred, output_dict=True)  
    
    def summary(self):
        tail = self.cls_history.tail(1)
        if self.approach == 'MARC':
            self.prediction()
            classification_report = metrics.compute_acc_acc5_f1_prec_rec(self.y_test_true, np.array(self.y_test_pred))
            tail2 = classification_report.tail(1)
            summ = {
                'acc':               tail['val_acc'].values,
                'acc_top_K5':        tail['val_top_k_categorical_accuracy'].values, 
                'balanced_accuracy': tail2['balanced_accuracy'].values,
                'precision_macro':   tail2['precision_macro'].values,
                'recall_macro':      tail2['recall_macro'].values,
                'f1_macro':          tail2['f1-macro'].values,
                'loss':              tail['val_loss'].values,
            }
        elif self.approach == 'POIS':
            summ = {
                'acc':               tail['test_acc'].values,
                'acc_top_K5':        tail['test_acc_top5'].values,
                'balanced_accuracy': None,
                'precision_macro':   tail['test_prec_macro'].values,
                'recall_macro':      tail['test_rec_macro'].values,
                'f1_macro':          tail['test_f1_macro'].values,
                'loss':              tail['test_loss'].values,
            }
        else:
            summ = {
                'acc':               tail['acc'].values,
                'acc_top_K5':        tail['acc_top_K5'].values,
                'balanced_accuracy': tail['balanced_accuracy'].values,
                'precision_macro':   tail['precision_macro'].values,
                'recall_macro':      tail['recall_macro'].values,
                'f1_macro':          tail['f1-macro'].values,
                'loss':              None,
            }
            
        
        return pd.DataFrame(summ)
        
    def save(self, dir_path, modelfolder):
        if not os.path.exists(os.path.join(dir_path, modelfolder)):
            os.makedirs(os.path.join(dir_path, modelfolder))
        self.classifier.save(os.path.join(dir_path, modelfolder, 'model_'+self.approach.lower()+'.h5'))

        prediction = self.prediction()
        prediction.to_csv(os.path.join(dir_path, modelfolder, 'model_'+self.approach.lower()+'_prediction.csv'), header = True) 
        
        report = self.report()
        classification_report_dict2csv(report, os.path.join(dir_path, modelfolder, 'model_'+self.approach.lower()+'_report.csv'), self.approach)            
        self.cls_history.to_csv(os.path.join(dir_path, modelfolder, 'model_'+self.approach.lower()+'_history.csv'))
