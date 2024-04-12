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
from matanalysis.methods._lib.datahandler import prepareTrajectories

from matanalysis.methods._lib.pymove.models.classification import DeepestST as DST
# --------------------------------------------------------------------------------

from matanalysis.methods.core import HPOClassifier

class DeepeST(HPOClassifier):
    
    def __init__(self, 
                 ## GRID SEARCH PARAMETERS
                 rnn = ['bilstm', 'lstm'],
                 units = [100, 200, 300, 400, 500],
                 merge_type = ['concat'],
                 dropout_before_rnn=[0, 0.5],
                 dropout_after_rnn=[0.5],

                 embedding_size = [50, 100, 200, 300, 400],
                 batch_size = [64],
                 epochs = [1000],
                 patience = [20],
                 monitor = ['val_acc'],

                 optimizer = ['ada'],
                 learning_rate = [0.001],
                 loss = ['CCE'],
                 loss_parameters = [{}], # TODO unfix, it´s fixed for now, but if you add parameters, change all configs.

                 y_one_hot_encodding = True,
                 
                 save_results=False,
                 n_jobs=-1,
                 verbose=0,
                 random_state=42,
                 filterwarnings='ignore'):
        
        super().__init__('DeepeST', save_results=save_results, n_jobs=n_jobs, verbose=verbose, random_state=random_state, filterwarnings=filterwarnings)
        
        self.add_config(rnn=rnn, 
                        units=units, 
                        merge_type=merge_type, 
                        dropout_before_rnn=dropout_before_rnn,
                        dropout_after_rnn=dropout_after_rnn, 
                        embedding_size=embedding_size, 
                        batch_size=batch_size, 
                        epochs=epochs, 
                        patience=patience, 
                        monitor=monitor, 
                        optimizer=optimizer, 
                        learning_rate=learning_rate,
                        loss=loss,
                        loss_parameters=loss_parameters,
                        y_one_hot_encodding=y_one_hot_encodding)

        # Moved to prepare_input:
#        self.grid = list(itertools.product())
        
        self.model = None
    
    def prepare_input(self,
                      train, test,
                      tid_col='tid', 
                      class_col='label',
                      space_geohash=False, # True: Geohash, False: indexgrid
                      geo_precision=30,     # Geohash: precision OR IndexGrid: meters
                      validate=True):
        
        ## Rewriting the method to change default params
        X, y, features, num_classes, space, dic_parameters = prepareTrajectories(train.copy(), test.copy(),
                                                                                 tid_col=tid_col, 
                                                                                 class_col=class_col,
                                                                                 # space_geohash, True: Geohash, False: indexgrid
                                                                                 space_geohash=space_geohash, 
                                                                                 # Geohash: precision OR IndexGrid: meters
                                                                                 geo_precision=geo_precision,   
                                                                                 
                                                                                 features_encoding=True, 
                                                                                 y_one_hot_encodding=True,
                                                                                 split_test_validation=validate,
                                                                                 data_preparation=2,
                                                                                 
                                                                                 verbose=self.isverbose)
        
        self.config['space'] = space
        self.config['dic_parameters'] = dic_parameters
        
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
        
        max_lenght  = self.config['max_lenght'] = dic_parameters['max_lenght']
        num_classes  = self.config['num_classes'] = dic_parameters['num_classes']
        vocab_size  = self.config['vocab_size'] = dic_parameters['vocab_size']
        features  = self.config['features'] = dic_parameters['features']
        encode_features  = self.config['encode_features'] = dic_parameters['encode_features']
        encode_y  = self.config['encode_y'] = dic_parameters['encode_y']

        ## GRID SEARCH PARAMETERS
        rnn = self.config['rnn']
        units = self.config['units']
        merge_type = self.config['merge_type']
        dropout_before_rnn = self.config['dropout_before_rnn']
        dropout_after_rnn = self.config['dropout_after_rnn']

        embedding_size = self.config['embedding_size']
        batch_size = self.config['batch_size']
        epochs = self.config['epochs']
        patience = self.config['patience']
        monitor = self.config['monitor']

        optimizer = self.config['optimizer']
        learning_rate = self.config['learning_rate']
        loss = self.config['loss']
        loss_parameters = self.config['loss_parameters']
        
        y_one_hot_encodding = self.config['y_one_hot_encodding']
        
        self.grid = list(itertools.product(rnn, units, merge_type, dropout_before_rnn, dropout_after_rnn, 
                                           embedding_size, batch_size, epochs, patience, monitor, optimizer, 
                                           learning_rate, loss, loss_parameters))
        
        return X, y, features, num_classes, space, dic_parameters
        
    def create(self, config):

        nn=config[0]
        un=config[1]
        mt=config[2]
        dp_bf=config[3]
        dp_af=config[4]
        em_s=config[5]
#        bs=config[6]
#        epoch=config[7] 
#        pat=config[8] 
#        mon=config[9] 
#        opt=config[10] 
#        lr=config[11]
#        ls=config[12]
#        ls_p=config[13]
        
        #Initializing Neural Network
        return DST.DeepeST(max_lenght=self.config['max_lenght'],
                           num_classes=self.config['num_classes'],
                           vocab_size=self.config['vocab_size'],
                           rnn=nn,
                           rnn_units=un,
                           merge_type = mt,
                           dropout_before_rnn=dp_bf,
                           dropout_after_rnn=dp_af,
                           embedding_size=em_s)
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
        
        bs=config[6]
        epoch=config[7] 
        pat=config[8] 
        mon=config[9] 
        opt=config[10] 
        lr=config[11]
        ls=config[12]
        ls_p=config[13]
        
        return self.model.fit(X_train, y_train,
                              X_val, y_val,
                              batch_size=bs,
                              epochs=epoch,
                              monitor=mon,
                              min_delta=0,
                              patience=pat,
                              verbose=0,
    #                          baseline=0.5,
                              baseline=None, # By Tarlis
                              optimizer=opt,
                              learning_rate=lr,
                              mode='auto',
                              new_metrics=None,
                              save_model=False,
                              modelname='',
                              save_best_only=True,
                              save_weights_only=False,
                              log_dir=None,
                              loss=ls,
                              loss_parameters=ls_p)
    
    def predict(self,                 
                X_test,
                y_test):
        
        self._summary, y_pred = self.model.predict(X_test, y_test)
        
        self.y_test_true = y_test
        self.y_test_pred = y_pred
        
        if self.le:
            self.y_test_true = self.le.inverse_transform(self.y_test_true).reshape(1, -1)[0]
            self.y_test_pred = self.le.inverse_transform(self.y_test_pred).reshape(1, -1)[0]
            
        return self._summary, y_pred 