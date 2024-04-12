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
from matanalysis.methods._lib.metrics import f1
from matanalysis.methods._lib.pymove.models import metrics
from sklearn.metrics import classification_report
from matanalysis.methods._lib.metrics import compute_acc_acc5_f1_prec_rec

from matanalysis.methods.core import MHPOClassifier

# Approach 2
class MMLP(MHPOClassifier):
    
    def __init__(self, 
                 nattr=-1,
                 nclasses=-1,
                 par_dropout = 0.5,
                 par_batch_size = 200,
#                 par_epochs = 80,
#                 par_lr = 0.00095,
                 lst_par_epochs = [80,50,50,30,20],
                 lst_par_lr = [0.00095,0.00075,0.00055,0.00025,0.00015],
                 n_jobs=-1,
                 verbose=2,
                 random_state=42,
                 filterwarnings='ignore'):
        super().__init__('MMLP', n_jobs=n_jobs, verbose=verbose, random_state=random_state, filterwarnings=filterwarnings)
        
        self.add_config(par_dropout=par_dropout, 
                        par_batch_size=par_batch_size, 
#                        par_epochs=par_epochs, 
#                        par_lr=par_lr, 
                        lst_par_epochs=lst_par_epochs, 
                        lst_par_lr=lst_par_lr, 
                        nattr=nattr, 
                        nclasses=nclasses)
        
    def create(self):
        
        nattr = self.config['num_features']
        nclasses = self.config['num_classes']
        par_dropout = self.config['par_dropout']
        
        #Initializing Neural Network
        self.model = Sequential()
        # Adding the input layer and the first hidden layer
        self.model.add(Dense(units = 100, kernel_initializer = 'uniform', activation = 'linear', input_dim = (nattr)))
        self.model.add(Dropout( par_dropout ))
        # Adding the output layer
        self.model.add(Dense(units = nclasses, kernel_initializer = 'uniform', activation = 'softmax'))
        
#    def prepare_input(self,
#                      train, test,
#                      validate = True):
#        
#        #Preparing the input of movelets:
#        X, y, nattr, num_classes = super().prepare_input(train, test, validate=validate)
#        
##        self.add_config(nattr=nattr, num_classes=num_classes)
#        
##        # Scaling y and transforming to keras format
##        self.le = preprocessing.LabelEncoder()
##        self.le.fit(self.y_train)
##        
##        if not self.validate:
##            self.y_train = self.le.transform(self.y_train) 
##            self.y_test = self.le.transform(self.y_test)
##            
##            self.y_train = to_categorical(self.y_train)
##            self.y_test = to_categorical(self.y_test)
##        else:
##            self.y_train = self.le.transform(self.y_train)
##            self.y_val = self.le.transform(self.y_val)
##            self.y_test = self.le.transform(self.y_test)
##            
##            self.y_train = to_categorical(self.y_train)
##            self.y_val = to_categorical(self.y_val)
##            self.y_test = to_categorical(self.y_test)
#            
#        return X, y, nattr, num_classes

        
    def fit(self, 
            X_train, 
            y_train, 
            X_val,
            y_val):

        self.config['num_features'] = len(X_train[1,:])  
        
#        # Scaling y and transforming to keras format
#        self.le = preprocessing.LabelEncoder()
#        self.le.fit(y_train)
#
#        y_train = self.le.transform(y_train) 
#        y_val = self.le.transform(y_val)
#
#        y_train1 = to_categorical(y_train)
#        y_val1 = to_categorical(y_val)
        self.config['num_classes'] = len(self.le.classes_)
        
        if not self.model:
            self.create()

        lst_par_lr = self.config['lst_par_lr']
        lst_par_epochs = self.config['lst_par_epochs']
        par_batch_size = self.config['par_batch_size']
        verbose=self.config['verbose']
        
        # Compiling Neural Network
        k = len(lst_par_epochs)

        for k in range(0,k) :
            adam = Adam(learning_rate=lst_par_lr[k])
            self.model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy','top_k_categorical_accuracy',f1])
            history = self.model.fit(X_train, y_train, validation_data = (X_val, y_val), epochs=lst_par_epochs[k], batch_size=par_batch_size, verbose=verbose)
        
        self.report = pd.DataFrame(history.history)
        return self.report
        
    def predict(self,                 
                X_test,
                y_test):
        
#        y_test = self.le.transform(y_test)
#        y_test1 = to_categorical(y_test)
        
        y_pred = self.model.predict(X_test) 
        
        self.y_test_true = self.le.inverse_transform(argmax(y_test, axis = 1))
        self.y_test_pred =  self.le.inverse_transform(argmax(y_pred , axis = 1)) 
    
        self.report = metrics.compute_acc_acc5_f1_prec_rec(self.y_test_true, self.y_test_pred)

#        acc, acc_top5, _f1_macro, _prec_macro, _rec_macro, bal_acc = compute_acc_acc5_f1_prec_rec(y_test1, np.array(y_pred)) #self.y_test_pred, y_pred)
#        
#        dic_model = {
#            'acc': acc,
#            'acc_top_K5': acc_top5,
#            'balanced_accuracy': bal_acc,
#            'precision_macro': _f1_macro,
#            'recall_macro': _prec_macro,
#            'f1_macro': _rec_macro,
#        } 
        
        self._summary = self.score(X_test, y_test, y_pred)
        
        if self.config['verbose']:
            print('['+self.name+':] Processing time: {} milliseconds. Done.'.format(self.duration()))
            
        return self._summary, y_pred

    
# --------------------------------------------------------------------------------
#Approach 1
class MMLP1(MHPOClassifier):
    
    def __init__(self, 
                 nattr=-1,
                 nclasses=-1,
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
                        nattr=nattr, 
                        nclasses=nclasses)
        
#    def prepare_input(self,
#                      train, test,
#                      validate = True):
#        
#        #Preparing the input of movelets:
#        X, y, nattr, num_classes = super().prepare_input(train, test, validate=validate)
#        
#        self.add_config(nattr=nattr, num_classes=num_classes)
#        
#        # Scaling y and transforming to keras format
#        self.le = preprocessing.LabelEncoder()
#        self.le.fit(self.y_train)
#        
#        if not self.validate:
#            self.y_train = self.le.transform(self.y_train) 
#            self.y_test = self.le.transform(self.y_test)
#            
#            self.y_train = to_categorical(self.y_train)
#            self.y_test = to_categorical(self.y_test)
#        else:
#            self.y_train = self.le.transform(self.y_train)
#            self.y_val = self.le.transform(self.y_val)
#            self.y_test = self.le.transform(self.y_test)
#            
#            self.y_train = to_categorical(self.y_train)
#            self.y_val = to_categorical(self.y_val)
#            self.y_test = to_categorical(self.y_test)
#            
#        return X, y, nattr, num_classes
        
    def create(self):
        
        nattr = self.config['num_features']
        nclasses = self.config['num_classes']
        par_dropout = self.config['par_dropout']
        par_lr = self.config['par_lr']
        
        #Initializing Neural Network
        self.model = Sequential()
        # Adding the input layer and the first hidden layer
        self.model.add(Dense(units = 100, kernel_initializer = 'uniform', kernel_regularizer= regularizers.l2(0.02), activation = 'relu', input_dim = (nattr)))
        #model.add(BatchNormalization())
        self.model.add(Dropout( par_dropout )) 
        # Adding the output layer       
        self.model.add(Dense(units = nclasses, kernel_initializer = 'uniform', activation = 'softmax'))
        # Compiling Neural Network
    #     adam = Adam(lr=par_lr) # TODO: check for old versions...
        adam = Adam(learning_rate=par_lr)
        self.model.compile(optimizer = adam, loss = 'categorical_crossentropy', metrics = ['accuracy','top_k_categorical_accuracy',f1])
        
        
        
        #assert (eval_metric == 'merror') | (eval_metric == 'mlogloss'), "ERR: invalid loss, set loss as mlogloss or merror" 

        #print('[MODEL:] Starting model training, {} iterations'.format(total))
        
    def fit(self, 
            X_train, 
            y_train, 
            X_val,
            y_val):

        self.config['num_features'] = len(X_train[1,:])  
        
#        # Scaling y and transforming to keras format
#        self.le = preprocessing.LabelEncoder()
#        self.le.fit(y_train)
#
#        y_train = self.le.transform(y_train) 
#        y_val = self.le.transform(y_val)
#
#        self.y_train = y_train1 = to_categorical(y_train)
#        self.y_val = y_val1 = to_categorical(y_val)
        self.config['num_classes'] = len(self.le.classes_)
        
        if not self.model:
            self.create()

        par_batch_size = self.config['par_batch_size']
        par_epochs = self.config['par_epochs']
        verbose=self.config['verbose']

        history = self.model.fit(X_train, y_train, validation_data = (X_val, y_val), batch_size = par_batch_size, epochs=par_epochs, verbose=verbose)
        
        self.report = pd.DataFrame(history.history)
        return self.report
        
    def predict(self,                 
                X_test,
                y_test):
        
#        y_test = self.le.transform(y_test)
#        self.y_text = y_test1 = to_categorical(y_test)
        
        y_pred = self.model.predict(X_test) 
        
        self.y_test_true = self.le.inverse_transform(argmax(y_test, axis = 1))
        self.y_test_pred =  self.le.inverse_transform(argmax(y_pred , axis = 1)) 
    
#        self._summary = metrics.compute_acc_acc5_f1_prec_rec(self.y_test_true, self.y_test_pred)
        self._summary = self.score(X_test, y_test, y_pred)
        
        if self.config['verbose']:
            print('['+self.name+':] Processing time: {} milliseconds. Done.'.format(self.duration()))
            
        return self._summary, self.y_test_pred