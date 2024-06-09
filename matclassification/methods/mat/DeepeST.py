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
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tensorflow.keras.layers import Dense, LSTM, GRU, Bidirectional, Concatenate, Add, Average, Embedding, Dropout, Input
from tensorflow.keras.initializers import he_normal, he_uniform
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.regularizers import l1
from tensorflow.keras import backend as K
# --------------------------------------------------------------------------------
from matclassification.methods._lib.datahandler import prepareTrajectories
from matclassification.methods.core import THSClassifier

class DeepeST(THSClassifier):
    
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
    
    def xy(self,
           train, test,
           tid_col='tid', 
           class_col='label',
           space_geohash=False, # True: Geohash, False: indexgrid
           geo_precision=30,    # Geohash: precision OR IndexGrid: meters
           validate=False):
        
        # RETURN: X, y, features, num_classes, space, dic_parameters
        return prepareTrajectories(train.copy(), test.copy(),
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
    
    def prepare_input(self,
                      train, test,
                      tid_col='tid', 
                      class_col='label',
                      space_geohash=False, # True: Geohash, False: indexgrid
                      geo_precision=30,     # Geohash: precision OR IndexGrid: meters
                      validate=False):
        
        ## Rewriting the method to change default params
        X, y, features, num_classes, space, dic_parameters = self.xy(train, test, tid_col, class_col, geo_precision, validate)
        
        self.add_config(space=space,
                        dic_parameters=dic_parameters)
        
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
        
        vocab_size=self.config['vocab_size']
        max_lenght=self.config['max_lenght']
        num_classes=self.config['num_classes'] # Tarlis
        col_name = list(vocab_size.keys())
        
        rnn=config[0]
        rnn_units=config[1]
        merge_type=config[2]
        dropout_before_rnn=config[3]
        dropout_after_rnn=config[4]
        embedding_size=config[5]
        
        input_model = []
        embedding_layers = []
        hidden_input = []
        hidden_dropout  = []
        
        if not isinstance(embedding_size, dict):
            embbeding_default = embedding_size
            embedding_size = dict(zip(col_name, np.full(len(col_name), embbeding_default)))
        
        assert set(vocab_size) == set(embedding_size), "ERR: embedding size is different from vocab_size"
        assert len(embedding_size) > 0, "embedding size was not defined"
        
        # Initializing Neural Network
        # Building Input and Embedding Layers
        for c in tqdm(col_name):
            i_model= Input(shape=(max_lenght,), 
                            name='Input_{}'.format(c)) 
            e_output_ = Embedding(input_dim = vocab_size[c], 
                                output_dim = embedding_size[c], 
                                name='Embedding_{}'.format(c), 
                                input_length=max_lenght)(i_model)

            input_model.append(i_model)  
            embedding_layers.append(e_output_)             

        # MERGE Layer
        if len(embedding_layers) == 1:
            hidden_input = embedding_layers[0]
        elif merge_type == 'add':
            hidden_input = Add()(embedding_layers)
        elif merge_type == 'avg':
            hidden_input = Average()(embedding_layers)
        else:
            hidden_input = Concatenate(axis=2)(embedding_layers)

        # DROPOUT before RNN
        hidden_dropout = Dropout(dropout_before_rnn)(hidden_input)
    
        # Recurrent Neural Network Layer
        # https://www.quora.com/What-is-the-meaning-of-%E2%80%9CThe-number-of-units-in-the-LSTM-cell
        if rnn == 'bilstm':
            rnn_cell = Bidirectional(LSTM(units=rnn_units, recurrent_regularizer=l1(0.02)))(hidden_dropout)
        else:
            rnn_cell = LSTM(units=rnn_units, recurrent_regularizer=l1(0.02))(hidden_dropout)

        rnn_dropout = Dropout(dropout_after_rnn)(rnn_cell)
        
        #https://keras.io/initializers/#randomnormal
        output_model = Dense(num_classes, 
                            kernel_initializer=he_uniform(),
                            activation='softmax')(rnn_dropout)

        # Encoding the labels as integers and using the sparse_categorical_crossentropy asloss function
        return Model(inputs=input_model, outputs=output_model)
    
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
        
        batch_size=config[6]
        epochs=config[7]
        monitor=config[9]
        min_delta=0 
        patience=config[8] 
        verbose=0
        baseline=None # By Tarlis
        optimizer=config[10]
        learning_rate=config[11]
        mode='auto'
        new_metrics=None
        modelname=''
        log_dir=None
        loss=config[12]
        loss_parameters=config[13]
        
        assert (y_train.ndim == 1) |  (y_train.ndim == 2), "ERR: y_train dimension is incorrect"            
        assert (y_val.ndim == 1) |  (y_val.ndim == 2), "ERR: y_test dimension is incorrect"
        assert (y_train.ndim == y_val.ndim), "ERR: y_train and y_test have different dimensions"

        if y_train.ndim == 1:
            y_one_hot_encodding = False
        elif y_train.ndim == 2:
            y_one_hot_encodding = True

        if y_one_hot_encodding == True:
            loss = ['categorical_crossentropy'] #categorical_crossentropy
            my_metrics = ['acc', 'top_k_categorical_accuracy'] 
        else:
            loss = ['sparse_categorical_crossentropy'] #sparse_categorical_crossentropy
            my_metrics = ['acc', 'sparse_top_k_categorical_accuracy']  

        # Tarlis: removed the top_k metric in cases of less than 5 classes
        if self.config['num_classes'] < 5:
            my_metrics = ['acc']

        if new_metrics is not None:
            my_metrics = new_metrics + my_metrics

        if optimizer == 'ada':
            # Optimizer was setting as Adam
            optimizer = Adam(lr=learning_rate)
        else:
            # Optimizer was setting as RMSProps
            optimizer = RMSprop(lr=learning_rate)

        # Compiling DeepeST Model
        self.model.compile(optimizer=optimizer, loss=loss, metrics=my_metrics)

        early_stop = EarlyStopping(monitor=monitor,
                                    min_delta=min_delta, 
                                    patience=patience, 
                                    verbose=verbose, # without print 
                                    mode=mode,
                                    baseline=baseline,
                                    restore_best_weights=True)


        # Defining checkpoint
        my_callbacks= [early_stop]    

        # Starting training
        return self.model.fit(X_train, y_train,
                            epochs=epochs,
                            callbacks=my_callbacks,
                            validation_data=(X_val, y_val),
                            verbose=1,
                            shuffle=True,
                            use_multiprocessing=True,          
                            batch_size=batch_size)
    
    def predict(self,                 
                X_test,
                y_test):
        
        assert (y_test.ndim == 1) |  (y_test.ndim == 2), "ERR: y_train dimension is incorrect"       

        if y_test.ndim == 1:
            y_one_hot_encodding = False
        elif y_test.ndim == 2:
            y_one_hot_encodding = True

        y_pred_prob = np.array(self.model.predict(X_test))
        
        if y_one_hot_encodding == True:
            argmax = np.argmax(y_pred_prob, axis=1)
            y_pred_true = np.zeros(y_pred_prob.shape)
            for row, col in enumerate(argmax):
                y_pred_true[row][col] = 1
        else:
            y_pred_true = y_pred_prob.argmax(axis=1)

        self._summary = self.score(np.argmax(y_test, axis=1), y_pred_prob)
        
        self.y_test_true = y_test
        self.y_test_pred = y_pred_true
        
        if self.le:
            self.y_test_true = self.le.inverse_transform(self.y_test_true).reshape(1, -1)[0]
            self.y_test_pred = self.le.inverse_transform(self.y_test_pred).reshape(1, -1)[0]
            
        return self._summary, y_pred_prob
    
    def clear(self):
        super().clear()
        K.clear_session()