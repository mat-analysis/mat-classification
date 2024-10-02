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
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tensorflow.keras.layers import Dense, Lambda, LSTM, GRU, Bidirectional, Concatenate, Add, Average, Embedding, Dropout, Input
from tensorflow.keras.initializers import he_normal, he_uniform
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, ConvLSTM2D, BatchNormalization, RepeatVector, Conv2D
from tensorflow.keras.regularizers import l1
from tensorflow.keras import backend as K
# --------------------------------------------------------------------------------
from matclassification.methods._lib.datahandler import prepareTrajectories
from matclassification.methods.core import THSClassifier

class Tulvae(THSClassifier):
    """
    Tulvae: Trajectory-user linking via variational autoencoder

    The `Tulvae` class is an implementation of a deep learning model based on variable auto-encoders, designed for trajectory classification tasks. It utilizes a variety of tunable hyperparameters and neural network structures to encode spatial trajectories and decode them for predictive modeling.

    Parameters
    ----------
    rnn : list, default=['bilstm']
        Recurrent neural network cell used, e.g., 'bilstm' (Bidirectional LSTM).
    units : list, default=[100, 200, 300]
        Number of units in the recurrent layers.
    stack : list, default=[1]
        Number of stacked recurrent layers.
    dropout : list, default=[0.5]
        Fraction of units to drop for the linear transformation of the inputs.
    embedding_size : list, default=[100, 200, 300]
        Size of the embedding vectors used to represent trajectory features.
    z_values : list, default=[100, 200, 300]
        Dimensionality of the latent variable space.
    batch_size : list, default=[64]
        Number of samples per batch of computation.
    epochs : list, default=[1000]
        Number of epochs to train the model.
    patience : list, default=[20]
        Number of epochs with no improvement after which training will be stopped.
    monitor : list, default=['val_acc']
        Metric used for early stopping and performance evaluation.
    optimizer : list, default=['ada']
        Optimizer used to minimize the loss function.
    learning_rate : list, default=[0.001]
        Learning rate for the optimizer.

    Methods
    -------
    xy(train, test, tid_col='tid', class_col='label', space_geohash=False, geo_precision=30, features=['poi'], validate=False):
        Prepares trajectory data and transforms it into training and testing datasets.
    prepare_input(train, test, tid_col='tid', class_col='label', space_geohash=False, geo_precision=30, features=['poi'], validate=False):
        Prepares the input data, generates configuration, and initializes grid search.
    create(config):
        Builds the neural network model based on the configuration.
    fit(X_train, y_train, X_val, y_val, config=None):
        Trains the model on the provided training data, using early stopping and validation data.
    clear():
        Clears the session and resets the model.
    """
    
    def __init__(self, 
#                 max_lenght = -1,
#                 vocab_size = -1,
                 rnn= ['bilstm'], #Unused
                 units = [100, 200, 300],
                 stack = [1],
                 dropout =[0.5],
                 embedding_size = [100, 200, 300],
                 z_values = [100,200,300],
                 batch_size = [64],
                 epochs = [1000],
                 patience = [20],
                 monitor = ['val_acc'],
                 optimizer = ['ada'],
                 learning_rate = [0.001],
                 
                 save_results=False,
                 n_jobs=-1,
                 verbose=0,
                 random_state=42,
                 filterwarnings='ignore'):
        
        super().__init__('Tulvae', save_results=save_results, n_jobs=n_jobs, verbose=verbose, random_state=random_state, filterwarnings=filterwarnings)
        
        self.add_config(rnn=rnn, 
                        units=units, 
                        stack=stack, 
                        dropout=dropout, 
                        embedding_size=embedding_size, 
                        z_values=z_values, 
                        batch_size=batch_size, 
                        epochs=epochs, 
                        patience=patience, #Unused
                        monitor=monitor, 
                        optimizer=optimizer, 
                        learning_rate=learning_rate)
        
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
        
        self.config['max_lenght'] = dic_parameters['max_lenght']
        self.config['vocab_size'] = dic_parameters['vocab_size'][features[0]] #['poi']
        rnn = self.config['rnn']
        units = self.config['units']
        stack = self.config['stack']
        dropout = self.config['dropout']
        embedding_size = self.config['embedding_size']
        z_values = self.config['z_values']
        batch_size = self.config['batch_size']
        epochs = self.config['epochs']
        patience = self.config['patience']
        monitor = self.config['monitor']
        optimizer = self.config['optimizer']
        learning_rate = self.config['learning_rate']
        
        self.grid_search(rnn, units, stack, dropout, embedding_size, z_values, batch_size,epochs, patience, monitor, learning_rate)
#        self.grid = list(itertools.product(rnn, units, stack, dropout, embedding_size, 
#                                           z_values, batch_size,epochs, patience, monitor, learning_rate))
        
        return X, y, features, num_classes, space, dic_parameters
        
    def create(self, config):
    
        max_lenght=self.config['max_lenght']
        num_classes=self.config['num_classes']
        vocab_size=self.config['vocab_size']
        rnn_units=config[1]
        stack=config[2]
        dropout=config[3]
        embedding_size=config[4]
        z_values=config[5]
        
        #Initializing Neural Network
        #### variables locals ##
        input_model = []
        embedding_layers = []
        hidden_input = []
        hidden_dropout  = []
        
        #Input
        input_model= Input(shape=(max_lenght,), name='spatial_poi')
        aux = RepeatVector(1)(input_model)

        # Embedding
        embedding_layer = Embedding(input_dim = vocab_size, output_dim = embedding_size,name='embedding_poi', input_length=max_lenght)(input_model)
        
        # Encoding
        encoder_lstm = Bidirectional(LSTM(rnn_units))(embedding_layer)
        encoder_lstm_dropout = Dropout(dropout)(encoder_lstm)

        # Latent
        z_mean = Dense(z_values)(encoder_lstm_dropout)
        z_log_sigma = Dense(z_values)(encoder_lstm_dropout)
        z = Lambda(sampling, output_shape=(z_values,))([z_mean, z_log_sigma,aux])

        # Decoding
        decoder_lstm = Bidirectional(LSTM(rnn_units))(RepeatVector(2)(z))
        decoder_lstm_dropout = Dropout(dropout)(decoder_lstm)

        #Output       
        output_model = Dense(num_classes, activation='softmax')(decoder_lstm_dropout)
    
        return Model(inputs=input_model, outputs=output_model)


    def fit(self, 
            X_train, 
            y_train, 
            X_val,
            y_val,
            config=None):
        
        if not config:
            config = self.best_config            
        if self.model == None:
            self.model = self.create(config)
            
        batch_size    = config[6]
        epochs        = config[7]
        learning_rate = config[10]
            
        ## seting parameters
        optimizer = Adam(lr=learning_rate)
        loss = ['sparse_categorical_crossentropy']
        metric = ['acc']  
        monitor='val_acc'

        self.model.compile(optimizer=optimizer, loss=loss, metrics=metric)

        early_stop = EarlyStopping(monitor='val_acc',
                                min_delta=0, 
                                patience=50, 
                                verbose=0, # without print 
                                mode='auto',
                                restore_best_weights=True)

       
        my_callbacks= [early_stop]   
        return self.model.fit(X_train, 
                            y_train,
                            epochs=epochs,
                            callbacks=my_callbacks,
                            validation_data=(X_val, y_val),
                            verbose=1,
                            shuffle=True,
                            use_multiprocessing=True,          
                            batch_size=batch_size)

    def clear(self):
        super().clear()
        K.clear_session()

def sampling_error(args):
    z_mean, z_log_sigma,aux = args
    bs = aux.shape[0]
    if(bs == None):
        epsilon = K.random_normal(shape=(1, 100),mean=0., stddev=1)
        return z_mean + z_log_sigma * epsilon
    else:
        epsilon = K.random_normal(shape=(bs, 100),mean=0., stddev=1)
        return z_mean + z_log_sigma * epsilon

def sampling(args):
    z_mean, z_log_sigma,aux = args
    bs = aux.shape[0]
    if(bs == None):
        epsilon = K.random_normal(shape=(1, z_mean.shape[1]),mean=0., stddev=1)
        return z_mean + z_log_sigma * epsilon
    else:
        epsilon = K.random_normal(shape=(bs, z_mean.shape[1]),mean=0., stddev=1)
        return z_mean + z_log_sigma * epsilon