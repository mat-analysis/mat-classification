# -*- coding: utf-8 -*-
'''
MAT-analysis: Analisys and Classification methods for Multiple Aspect Trajectory Data Mining

The present package offers a tool, to support the user in the task of data analysis of multiple aspect trajectories. It integrates into a unique framework for multiple aspects trajectories and in general for multidimensional sequence data mining methods.
Copyright (C) 2022, MIT license (this portion of code is subject to licensing from source project distribution)

Created on Dec, 2021
Copyright (C) 2022, License GPL Version 3 or superior (see LICENSE file)

@author: Tarlis Portela
@author: Francisco Vicenzi (adapted)
'''
# --------------------------------------------------------------------------------
import os
from os import path
import pandas as pd
import numpy as np

from tqdm.auto import tqdm
from datetime import datetime
# --------------------------------------------------------------------------------
from tensorflow import random
from numpy import argmax

from sklearn.preprocessing import scale, OneHotEncoder

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, History

from tensorflow.keras import backend as K

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from matanalysis.methods.pois.poifreq import pois

from matanalysis.methods._lib.datahandler import loadTrajectories
from matanalysis.methods._lib.pymove.models import metrics
from matanalysis.methods._lib.models import ModelContainer

## --------------------------------------------------------------------------------------------
## CLASSIFIER:
### Run Before: pois(df_train, df_test, sequences, features)
## --------------------------------------------------------------------------------------------
def POIS_read(dir_path, sequences, features, dataset='specific', method='npoi', res_path='.', prefix='', save_results=False, n_jobs=-1, random_state=42, rounds=10, tid_col='tid', class_col='label'):
#    importer(['S', 'POIS', 'random', 'datetime'], globals())

    # Load Data - Tarlis:
    df_train, df_test = loadTrajectories(dir_path, prefix)
    
    return POIS(df_train, df_test, sequences, features, dataset, method, res_path, prefix, save_results, n_jobs, random_state, rounds, tid_col, class_col)

def POIS(df_train, df_test, sequences, features, dataset='specific', method='npoi', res_path='.', prefix='', save_results=False, n_jobs=-1, random_state=42, rounds=10, tid_col='tid', class_col='label'):
    
    x_train, x_test, y_train, y_test, _ = pois(df_train, df_test, sequences, features, method, dataset, res_path, save_results, tid_col, class_col)
    
    return pois_model(x_train, x_test, y_train, y_test, method, res_path, prefix, save_results, n_jobs, random_state, rounds)
#    
#    y_pred = classifier.predict(x_test) 
#    final_pred = [argmax(f) for f in y_pred]
#    final_pred = [labels[f] for f in final_pred]
#    
#    return classification_report

def POIS_xy(dir_path, method='npoi', res_path='.', prefix='', save_results=False, n_jobs=-1, random_state=42, rounds=10):
#    importer(['S', 'POIS', 'random', 'datetime'], globals())

    x_train, x_test, y_train, y_test = loadData(dir_path)
    
    return pois_model(x_train, x_test, y_train, y_test, method, res_path, prefix, save_results, n_jobs, random_state, rounds)
#
#    y_pred = classifier.predict(x_test) 
#    final_pred = [argmax(f) for f in y_pred]
#    final_pred = [labels[f] for f in final_pred]
#    
#    classification_report = metrics.compute_acc_acc5_f1_prec_rec(y_test, np.array(final_pred))
#    return classification_report

## --------------------------------------------------------------------------------------------
def pois_model(x_train, x_test, y_train, y_test, method='npoi', res_path='.', prefix='', save_results=False, n_jobs=-1, random_state=42, rounds=10):
#    importer(['S', 'POIS', 'random', 'datetime'], globals())
# TODO: n_jobs=-1, rounds=10, geohash=False, geo_precision=30
# TODO: Transform into a model class

    (num_features, num_classes, labels, x_train, x_test, y_train, y_test, one_hot_y) = prepareData(x_train, x_test, y_train, y_test)

    np.random.seed(random_state)
    random.set_seed(random_state)

    keep_prob = 0.5

    HIDDEN_UNITS = 100
    LEARNING_RATE = 0.001
    EARLY_STOPPING_PATIENCE = 30
    BASELINE_METRIC = 'acc'
    BASELINE_VALUE = 0.5
    BATCH_SIZE = 64
    EPOCHS = 250
    #method = 'npoi'
    
#     labels = y_test

    print('[POI-S:] Building Neural Network')
    time = datetime.now()
    
    metrics_file = os.path.join(res_path, 'POIFREQ-metrics.csv')

    if save_results:
        if not os.path.exists(os.path.join(os.path.dirname(metrics_file))):
            os.makedirs(os.path.join(os.path.dirname(metrics_file)))
    metrics = MetricsLogger().load(metrics_file)

    print('keep_prob =', keep_prob)
    print('HIDDEN_UNITS =', HIDDEN_UNITS)
    print('LEARNING_RATE =', LEARNING_RATE)
    print('EARLY_STOPPING_PATIENCE =', EARLY_STOPPING_PATIENCE)
    print('BASELINE_METRIC =', BASELINE_METRIC)
    print('BASELINE_VALUE =', BASELINE_VALUE)
    print('BATCH_SIZE =', BATCH_SIZE)
    print('EPOCHS =', EPOCHS, '\n')


    class EpochLogger(EarlyStopping):

        def __init__(self, metric='val_acc', baseline=0):
            super(EpochLogger, self).__init__(monitor='val_acc',
                                              mode='max',
                                              patience=EARLY_STOPPING_PATIENCE)
            self._metric = metric
            self._baseline = baseline
            self._baseline_met = False

        def on_epoch_begin(self, epoch, logs={}):
            print("===== Training Epoch %d =====" % (epoch + 1))

            if self._baseline_met:
                super(EpochLogger, self).on_epoch_begin(epoch, logs)

        def on_epoch_end(self, epoch, logs={}):
            pred_y_train = np.array(self.model.predict(x_train))
            (train_acc,
             train_acc5,
             train_f1_macro,
             train_prec_macro,
             train_rec_macro) = compute_acc_acc5_f1_prec_rec(y_train,
                                                             pred_y_train,
                                                             print_metrics=True,
                                                             print_pfx='TRAIN')

            pred_y_test = np.array(self.model.predict(x_test))
            (test_acc,
             test_acc5,
             test_f1_macro,
             test_prec_macro,
             test_rec_macro) = compute_acc_acc5_f1_prec_rec(y_test, pred_y_test,
                                                            print_metrics=True,
                                                            print_pfx='TEST')
            metrics.log(method, int(epoch + 1), '',
                        logs['loss'], train_acc, train_acc5,
                        train_f1_macro, train_prec_macro, train_rec_macro,
                        logs['val_loss'], test_acc, test_acc5,
                        test_f1_macro, test_prec_macro, test_rec_macro)
            if save_results:
                metrics.save(metrics_file)

            if self._baseline_met:
                super(EpochLogger, self).on_epoch_end(epoch, logs)

            if not self._baseline_met \
               and logs[self._metric] >= self._baseline:
                self._baseline_met = True

        def on_train_begin(self, logs=None):
            super(EpochLogger, self).on_train_begin(logs)

        def on_train_end(self, logs=None):
            if self._baseline_met:
                super(EpochLogger, self).on_train_end(logs)

    classifier = Sequential()
    hist = History()
    classifier.add(Dense(units=HIDDEN_UNITS,
                    input_dim=(num_features),
                    kernel_initializer='uniform',
                    kernel_regularizer=l2(0.02)))
    classifier.add(Dropout(keep_prob))
    classifier.add(Dense(units=num_classes,
                    kernel_initializer='uniform',
                    activation='softmax'))

    opt = Adam(lr=LEARNING_RATE)
    classifier.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['acc'])

    classifier.fit(x=x_train,
              y=y_train,
              validation_data=(x_test, y_test),
              batch_size=BATCH_SIZE,
              shuffle=True,
              epochs=EPOCHS,
              verbose=0,
              callbacks=[EpochLogger(metric=BASELINE_METRIC,
                                     baseline=BASELINE_VALUE), hist])
    
    #print(metrics_file)
    time_ext = (datetime.now()-time).total_seconds() * 1000
    print("[POI-S:] Processing time: {time_ext} milliseconds. Done.")
    print('------------------------------------------------------------------------------------------------')
    
#    if save_results:
#        results_file = os.path.join(res_path, method+'_results.txt')
#
#        f = open(results_file, 'a+')
#        print('------------------------------------------------------------------------------------------------', file=f)
#    #     print(f"method: {method} | Dataset: {DATASET}", file=f)
#        print(f"Acc: {np.array(df['test_acc'])[-EARLY_STOPPING_PATIENCE]} ", file=f)
#        print(f"Acc_top_5: {np.array(df['test_acc_top5'])[-EARLY_STOPPING_PATIENCE]} ", file=f)
#        print(f"F1_Macro: {np.array(df['test_f1_macro'])[-EARLY_STOPPING_PATIENCE]} ", file=f)
#        print(f"Precision_Macro: {np.array(df['test_prec_macro'])[-EARLY_STOPPING_PATIENCE]} ", file=f)
#        print(f"Recall_Macro: {np.array(df['test_rec_macro'])[-EARLY_STOPPING_PATIENCE]} ", file=f)
#    
#        print(f"Processing time: {time_ext} milliseconds. Done.", file=f)
#        print('------------------------------------------------------------------------------------------------', file=f)
#        f.close()
    
#     print(labels)
    
#     return classifier.predict(x_test)
#    return classifier, x_test, labels

    # ---------------------------------------------------------------------------------
    # Prediction
    # ---------------------------------------------------------------------------------
    model = ModelContainer(classifier, y_test, x_test, cls_history=metrics._df, approach='POIS', le=one_hot_y)
    
    if save_results:
        model.save(dir_path, modelfolder)
    
    return model
# --------------------------------------------------------------------------------------------------------  
def loadData(dir_path):
#     from ..main import importer
#    importer(['S', 'PP', 'OneHotEncoder'], globals())
#     from sklearn.preprocessing import OneHotEncoder
#     from sklearn import preprocessing

    x_train = pd.read_csv(dir_path+'-x_train.csv', header=None)
    # x_train = x_train[x_train.columns[:-1]]
    y_train = pd.read_csv(dir_path+'-y_train.csv')

    x_test = pd.read_csv(dir_path+'-x_test.csv', header=None)
    # x_test = x_test[x_test.columns[:-1]]
    y_test = pd.read_csv(dir_path+'-y_test.csv')
    
    return x_train, x_test, y_train, y_test
    
# --------------------------------------------------------------------------------------------------------    
def prepareData(x_train, x_test, y_train, y_test):
    
#    labels = list(y_test[class_col])
    labels = list(y_test)

    num_features = len(list(x_train))
#    num_classes = len(y_train[class_col].unique())
    num_classes = len(set(y_train))
    
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    one_hot_y = OneHotEncoder()
#    one_hot_y.fit(y_train.loc[:, [class_col]])
    one_hot_y.fit(y_train)

#    y_train = one_hot_y.transform(y_train.loc[:, [class_col]]).toarray()
#    y_test = one_hot_y.transform(y_test.loc[:, [class_col]]).toarray()
    y_train = one_hot_y.transform(y_train).toarray()
    y_test = one_hot_y.transform(y_test).toarray()

    x_train = scale(x_train)
    x_test = scale(x_test)
    
    return (num_features, num_classes, labels, x_train, x_test, y_train, y_test, one_hot_y)


## --------------------------------------------------------------------------------------------
# from ..main import importer
#importer(['metrics', 'datetime'], globals())


def _process_pred(y_pred):
    argmax = np.argmax(y_pred, axis=1)
    y_pred = np.zeros(y_pred.shape)

    for row, col in enumerate(argmax):
        y_pred[row][col] = 1

    return y_pred


def f1_tensorflow_macro(y_true, y_pred):
    print(K.eval(y_pred))
    print(y_pred.shape)
    y_pred = np.zeros(y_pred.shape)
    for row, col in enumerate(argmax):
        y_pred[row][col] = 1
    
    # proc_y_pred = _process_pred(y_pred)
    return f1_score(y_true, y_pred, average='macro')


def precision_macro(y_true, y_pred):
    proc_y_pred = _process_pred(y_pred)
    return precision_score(y_true, proc_y_pred, average='macro')


def recall_macro(y_true, y_pred):
    proc_y_pred = _process_pred(y_pred)
    return recall_score(y_true, proc_y_pred, average='macro')


def f1_macro(y_true, y_pred):
    proc_y_pred = _process_pred(y_pred)
    return f1_score(y_true, proc_y_pred, average='macro')


def accuracy(y_true, y_pred):
    proc_y_pred = _process_pred(y_pred)
    return accuracy_score(y_true, proc_y_pred, normalize=True)


def accuracy_top_k(y_true, y_pred, K=5):
    order = np.argsort(y_pred, axis=1)
    correct = 0

    for i, sample in enumerate(np.argmax(y_true, axis=1)):
        if sample in order[i, -K:]:
            correct += 1

    return correct / len(y_true)


def compute_acc_acc5_f1_prec_rec(y_true, y_pred, print_metrics=True,
                                 print_pfx=''):
    acc = accuracy(y_true, y_pred)
    acc_top5 = accuracy_top_k(y_true, y_pred, K=5)
    _f1_macro = f1_macro(y_true, y_pred)
    _prec_macro = precision_macro(y_true, y_pred)
    _rec_macro = recall_macro(y_true, y_pred)

    if print_metrics:
        pfx = '' if print_pfx == '' else print_pfx + '\t\t'
        print(pfx + 'acc: %.6f\tacc_top5: %.6f\tf1_macro: %.6f\tprec_macro: %.6f\trec_macro: %.6f'
              % (acc, acc_top5, _f1_macro, _prec_macro, _rec_macro))

    return acc, acc_top5, _f1_macro, _prec_macro, _rec_macro


class MetricsLogger:

    def __init__(self):
        self._df = pd.DataFrame({'method': [],
                                 'epoch': [],
                                 'dataset': [],
                                 'timestamp': [],
                                 'train_loss': [],
                                 'train_acc': [],
                                 'train_acc_top5': [],
                                 'train_f1_macro': [],
                                 'train_prec_macro': [],
                                 'train_rec_macro': [],
                                 'train_acc_up': [],
                                 'test_loss': [],
                                 'test_acc': [],
                                 'test_acc_top5': [],
                                 'test_f1_macro': [],
                                 'test_prec_macro': [],
                                 'test_rec_macro': [],
                                 'test_acc_up': []})

    def log(self, method, epoch, dataset, train_loss, train_acc,
            train_acc_top5, train_f1_macro, train_prec_macro, train_rec_macro,
            test_loss, test_acc, test_acc_top5, test_f1_macro,
            test_prec_macro, test_rec_macro):
        timestamp = datetime.now()

        if len(self._df) > 0:
            train_max_acc = self._df['train_acc'].max()
            test_max_acc = self._df['test_acc'].max()
        else:
            train_max_acc = 0
            test_max_acc = 0

        self._df = self._df.append({'method': method,
                                    'epoch': epoch,
                                    'dataset': dataset,
                                    'timestamp': timestamp,
                                    'train_loss': train_loss,
                                    'train_acc': train_acc,
                                    'train_acc_top5': train_acc_top5,
                                    'train_f1_macro': train_f1_macro,
                                    'train_prec_macro': train_prec_macro,
                                    'train_rec_macro': train_rec_macro,
                                    'train_acc_up': 1 if train_acc > train_max_acc else 0,
                                    'test_loss': test_loss,
                                    'test_acc': test_acc,
                                    'test_acc_top5': test_acc_top5,
                                    'test_f1_macro': test_f1_macro,
                                    'test_prec_macro': test_prec_macro,
                                    'test_rec_macro': test_rec_macro,
                                    'test_acc_up': 1 if test_acc > test_max_acc else 0},
                                   ignore_index=True)

    def save(self, file):
        self._df.to_csv(file, index=False)

    def load(self, file):
        if os.path.isfile(file):
            self._df = pd.read_csv(file)
        else:
            print("WARNING: File '" + file + "' not found!")

        return self