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

from tqdm.auto import tqdm
# --------------------------------------------------------------------------------
from sklearn import preprocessing
# --------------------------------------------------------------------------------
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.utils import to_categorical
from matclassification.methods._lib.metrics import f1
#from matclassification.methods._lib.pymove.models import metrics
from sklearn.metrics import classification_report
from matclassification.methods._lib.metrics import compute_acc_acc5_f1_prec_rec

from matclassification.methods.core import MHSClassifier

# Approach 2
class MMLP(MHSClassifier):
    
    def __init__(self, 
                 num_features=-1,
                 num_classes=-1,
                 par_dropout = 0.5,
                 par_batch_size = 200,
#                 lst_par_epochs = [80,50,50,30,20],
#                 lst_par_lr = [0.00095,0.00075,0.00055,0.00025,0.00015],
                 lst_par_epochs_lr=[
                     [80, 0.00095],
                     [50, 0.00075],
                     [50, 0.00055],
                     [30, 0.00025],
                     [20, 0.00015],
                 ],
                 n_jobs=-1,
                 verbose=2,
                 random_state=42,
                 filterwarnings='ignore'):
        super().__init__('MMLP', n_jobs=n_jobs, verbose=verbose, random_state=random_state, filterwarnings=filterwarnings)
        
        self.add_config(par_dropout=par_dropout, 
                        par_batch_size=par_batch_size, 
#                        lst_par_epochs=lst_par_epochs, 
#                        lst_par_lr=lst_par_lr, 
                        lst_par_epochs_lr=lst_par_epochs_lr,
                        num_features=num_features, 
                        num_classes=num_classes)
        
    def create(self):
        
        nattr = self.config['num_features']
        nclasses = self.config['num_classes']
        par_dropout = self.config['par_dropout']
        
        #Initializing Neural Network
        model = Sequential()
        # Adding the input layer and the first hidden layer
        model.add(Dense(units = 100, kernel_initializer = 'uniform', activation = 'linear', input_dim = (nattr)))
        model.add(Dropout( par_dropout ))
        # Adding the output layer
        model.add(Dense(units = nclasses, kernel_initializer = 'uniform', activation = 'softmax'))
        
        return model
        
    def fit(self, 
            X_train, 
            y_train, 
            X_val,
            y_val):

        if self.model == None:
            self.model = self.create()

#        lst_par_lr = self.config['lst_par_lr']
#        lst_par_epochs = self.config['lst_par_epochs']
        lst_par_epochs_lr = self.config['lst_par_epochs_lr']
        par_batch_size = self.config['par_batch_size']
        verbose=self.config['verbose']
        
        # Compiling Neural Network
#        k = len(lst_par_epochs)

#        for k in range(0,k) :
        for epoch, lr in lst_par_epochs_lr:
            adam = Adam(learning_rate=lr)
            self.model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy','top_k_categorical_accuracy',f1])
            history = self.model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epoch, batch_size=par_batch_size, verbose=verbose)
        
        self.report = pd.DataFrame(history.history)
        return self.report
        
    def predict(self,                 
                X_test,
                y_test):
        
        y_pred = self.model.predict(X_test) 
        
        y_test_true = argmax(y_test, axis = 1)
        y_test_pred = argmax(y_pred , axis = 1)
        
        self._summary = self.score(y_test_true, y_pred)
        
        if self.le:
            self.y_test_true = self.le.inverse_transform(y_test_true)
            self.y_test_pred =  self.le.inverse_transform(y_test_pred) 
    
        return self._summary, y_pred

    
# --------------------------------------------------------------------------------
#Approach 1
class MMLP1(MHSClassifier):
    
    def __init__(self, 
                 num_features=-1,
                 num_classes=-1,
                 par_dropout = 0.5,
                 par_batch_size = 200,
                 par_epochs = 80,
                 par_lr = 0.00095,
#                 lst_par_epochs = [80,50,50,30,20],
#                 lst_par_lr = [0.00095,0.00075,0.00055,0.00025,0.00015],
                 n_jobs=-1,
                 verbose=2,
                 random_state=42,
                 filterwarnings='ignore'):
        
        super().__init__('MMLP1', n_jobs=n_jobs, verbose=verbose, random_state=random_state, filterwarnings=filterwarnings)
        
        self.add_config(par_dropout=par_dropout, 
                        par_batch_size=par_batch_size, 
                        par_epochs=par_epochs, 
                        par_lr=par_lr, 
#                        lst_par_epochs=lst_par_epochs, 
#                        lst_par_lr=lst_par_lr, 
                        num_features=num_features, 
                        num_classes=num_classes)
        
    def create(self):
        
        nattr = self.config['num_features']
        nclasses = self.config['num_classes']
        par_dropout = self.config['par_dropout']
        par_lr = self.config['par_lr']
        
        #Initializing Neural Network
        model = Sequential()
        # Adding the input layer and the first hidden layer
        model.add(Dense(units = 100, kernel_initializer = 'uniform', kernel_regularizer= regularizers.l2(0.02), activation = 'relu', input_dim = (nattr)))
        #model.add(BatchNormalization())
        model.add(Dropout( par_dropout )) 
        # Adding the output layer       
        model.add(Dense(units = nclasses, kernel_initializer = 'uniform', activation = 'softmax'))
        # Compiling Neural Network
    #     adam = Adam(lr=par_lr) # TODO: check for old versions...
        adam = Adam(learning_rate=par_lr)
        model.compile(optimizer = adam, loss = 'categorical_crossentropy', metrics = ['accuracy','top_k_categorical_accuracy',f1])
        
        return model
    
    def fit(self, 
            X_train, 
            y_train, 
            X_val,
            y_val):

        self.config['num_features'] = len(X_train[1,:])  
        self.config['num_classes'] = len(self.le.classes_)
        
        if not self.model:
            self.model = self.create()

        par_batch_size = self.config['par_batch_size']
        par_epochs = self.config['par_epochs']
        verbose=self.config['verbose']

        history = self.model.fit(X_train, y_train, validation_data = (X_val, y_val), batch_size = par_batch_size, epochs=par_epochs, verbose=verbose)
        
        self.report = pd.DataFrame(history.history)
        return self.report

    def predict(self,                 
                X_test,
                y_test):
        
        y_pred = self.model.predict(X_test) 
        
        y_test_true = argmax(y_test, axis = 1)
        y_test_pred = argmax(y_pred , axis = 1)
        
        self._summary = self.score(y_test_true, y_pred)
        
        if self.le:
            self.y_test_true = self.le.inverse_transform(y_test_true)
            self.y_test_pred =  self.le.inverse_transform(y_test_pred) 
            
        return self._summary, y_pred