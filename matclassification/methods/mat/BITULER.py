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
from matclassification.methods._lib.datahandler import prepareTrajectories

from matclassification.methods._lib.pymove.models.classification import Tuler as tul
# --------------------------------------------------------------------------------

from matclassification.methods.core import HPOClassifier

class BITULER(HPOClassifier):
    
    def __init__(self, 
#                 num_classes = -1,
#                 max_lenght = -1,
#                 vocab_size = -1,
                 rnn= ['bilstm'], #Unused
                 units = [100, 200, 250, 300],
                 stack = [1],
                 dropout =[0.5],
                 embedding_size = [100, 200, 300, 400],
                 batch_size = [64],
                 epochs = [1000],
                 patience = [20], #Unused
                 monitor = ['val_acc'],
                 optimizer = ['ada'],
                 learning_rate = [0.001],
                 
                 save_results=False,
                 n_jobs=-1,
                 verbose=0,
                 random_state=42,
                 filterwarnings='ignore'):
        
        super().__init__('TXGB', save_results=save_results, n_jobs=n_jobs, verbose=verbose, random_state=random_state, filterwarnings=filterwarnings)
        
        self.add_config(rnn=rnn, 
                        units=units, 
                        stack=stack, 
                        dropout=dropout, 
                        embedding_size=embedding_size, 
                        batch_size=batch_size, 
                        epochs=epochs, 
                        patience=patience, 
                        monitor=monitor, 
                        optimizer=optimizer, 
                        learning_rate=learning_rate)

        # Moved to prepare_input:
#        self.grid = list(itertools.product())
        
        self.model = None
    
    def xy(self,
           train, test,
           tid_col='tid', 
           class_col='label',
           space_geohash=False, # True: Geohash, False: indexgrid
           geo_precision=30,    # Geohash: precision OR IndexGrid: meters
           features=['poi'],
           validate=False):
        
        # RETURN: X, y, features, num_classes, space, dic_parameters
        return prepareTrajectories(train.copy(), test.copy(),
                                   tid_col=tid_col, 
                                   class_col=class_col,
                                   # space_geohash, True: Geohash, False: indexgrid
                                   space_geohash=space_geohash, 
                                   # Geohash: precision OR IndexGrid: meters
                                   geo_precision=geo_precision,     

                                   features=features,
                                   features_encoding=True, 
                                   y_one_hot_encodding=False,
                                   split_test_validation=validate,
                                   data_preparation=2,

                                   verbose=self.isverbose)
    
    def prepare_input(self,
                      train, test,
                      tid_col='tid', 
                      class_col='label',
                      space_geohash=False, # True: Geohash, False: indexgrid
                      geo_precision=30,     # Geohash: precision OR IndexGrid: meters
                      features=['poi'],
                      validate=False):
        
        ## Rewriting the method to change default params
        X, y, features, num_classes, space, dic_parameters = self.xy(train, test, tid_col, class_col, space_geohash, geo_precision, features, validate)
        
        self.add_config(space=space,
                        features=features,
                        num_classes=num_classes, 
                        dic_parameters=dic_parameters)
#        self.config['space'] = space
#        self.config['dic_parameters'] = dic_parameters
#        self.config['num_classes'] = num_classes
        
        if 'encode_y' in dic_parameters.keys():
            self.le = dic_parameters['encode_y']
        
        if len(X) == 2:
            self.X_train = X[0] 
            self.X_test = X[1]
            self.y_train = y[0] 
            self.y_test = y[1]
            self.validate = False
        if len(X) > 2:
            self.X_train = X[0] 
            self.X_val = X[1]
            self.X_test = X[2]
            self.y_train = y[0] 
            self.y_val = y[1]
            self.y_test = y[2]
            self.validate = True
        
        num_classes = self.config['num_classes'] = dic_parameters['num_classes']
        max_lenght = self.config['max_lenght'] = dic_parameters['max_lenght']
        vocab_size = self.config['vocab_size'] = dic_parameters['vocab_size'][features[0]] #['poi']
        rnn = self.config['rnn']
        units = self.config['units']
        stack = self.config['stack']
        dropout = self.config['dropout']
        embedding_size = self.config['embedding_size']
        batch_size = self.config['batch_size']
        epochs = self.config['epochs']
        patience = self.config['patience']
        monitor = self.config['monitor']
        optimizer = self.config['optimizer']
        learning_rate = self.config['learning_rate']
        
        self.grid = list(itertools.product(rnn, units, stack, dropout, embedding_size, 
                                           batch_size, epochs, patience, monitor, learning_rate))
        
        return X, y, features, num_classes, space, dic_parameters
        
    def create(self, config):

#        nn=config[0]
        un=config[1]
        st=config[2]
        dp=config[3]
        es=config[4]
#        bs=config[5]
#        epoch=config[6]
#        pat=config[7]
#        mon=config[8]
#        lr=config[9]
        
        #Initializing Neural Network
        return tul.BiTulerLSTM(max_lenght=self.config['max_lenght'],    
                               num_classes=self.config['num_classes'],
                               vocab_size=self.config['vocab_size'],
                               rnn_units=un,
                               dropout=dp,
                               embedding_size=es,
                               stack=st)
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
        
        bs=bs=config[5]
        epoch=config[6]
        lr=config[9]
        
        return self.model.fit(X_train, y_train,
                              X_val, y_val,
                              batch_size=bs,
                              epochs=epoch,
                              learning_rate=lr,
                              save_model=False,
                              save_best_only=False,
                              save_weights_only=False)
