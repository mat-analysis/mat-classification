# -*- coding: utf-8 -*-
'''
MAT-analysis: Analisys and Classification methods for Multiple Aspect Trajectory Data Mining

The present package offers a tool, to support the user in the task of data analysis of multiple aspect trajectories. It integrates into a unique framework for multiple aspects trajectories and in general for multidimensional sequence data mining methods.

Created on Dec, 2021
Copyright (C) 2022, License GPL Version 3 or superior (see LICENSE file)

@author: Tarlis Portela
@author: Lucas May Petry (adapted)
@author: Francisco Vicenzi (adapted)
'''
# --------------------------------------------------------------------------------
import os
import numpy as np
import pandas as pd

#from main import importer #, display
#importer(['S', 'datetime', 'metrics', 'K'], globals())

from datetime import datetime

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, top_k_accuracy_score, balanced_accuracy_score

from tensorflow.keras import backend as K
# --------------------------------------------------------------------------------

def _process_pred(y_pred):
    argmax = np.argmax(y_pred, axis=1)
    y_pred = np.zeros(y_pred.shape)

    for row, col in enumerate(argmax):
        y_pred[row][col] = 1

    return y_pred

#def expand_y(y_pred):
#    nlabels = len(set(y_pred))
#    proc_y_pred = np.zeros((len(y_pred), nlabels))
#
#    for row, col in enumerate(y_pred):
#        proc_y_pred[row][col] = 1
#
#    return proc_y_pred

#def f1_tensorflow_macro(y_true, y_pred):
#    y_pred = np.zeros(y_pred.shape)
#    for row, col in enumerate(argmax):
#        y_pred[row][col] = 1
#    
#    return f1_score(y_true, y_pred, average='macro')

#def precision_macro(y_true, y_pred):
#    proc_y_pred = _process_pred(y_pred)
#    return precision_score(y_true, proc_y_pred, average='macro', zero_division=1)
def precision_macro(y_true, y_pred):
    return precision_score(y_true, y_pred, average='macro', zero_division=1)

#def recall_macro(y_true, y_pred):
#    proc_y_pred = _process_pred(y_pred)
#    return recall_score(y_true, proc_y_pred, average='macro')
def recall_macro(y_true, y_pred):
    return recall_score(y_true, y_pred, average='macro')

#def f1_macro(y_true, y_pred):
#    proc_y_pred = _process_pred(y_pred)
#    return f1_score(y_true, proc_y_pred, average='macro')
def f1_macro(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')

#def accuracy(y_true, y_pred):
#    proc_y_pred = _process_pred(y_pred)
#    return accuracy_score(y_true, proc_y_pred, normalize=True)
def accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred, normalize=True)

# TODO: should be replaced for a standard function
def accuracy_top_k(y_true, y_pred, K=5):
    if y_pred.ndim == 1: # Not possible to calculate acc top K
        return 0
    else:
        order = np.argsort(y_pred, axis=1)
        correct = 0
        
#        for i, sample in enumerate(np.argmax(y_true, axis=1)):
        for i, sample in enumerate(y_true):
            if sample in order[i, -K:]:
                correct += 1

        return correct / len(y_true)

## by Tarlis --
#def acc_top_k(y_true, y_pred, n_y_pred, K ):
#    order = np.argsort(n_y_pred, axis=1)
#    correct = 0
#
#    for i, sample in enumerate(np.argmax(y_true, axis=1)):
#        if sample in order[i, -K:]:
#            correct += 1
#
#    return correct / len(y_true)

def balanced_accuracy(y_true, y_pred):
    if y_pred.ndim == 1:
        return balanced_accuracy_score(y_true, y_pred)
    else:
        return balanced_accuracy_score(y_true.argmax(axis=1), y_pred.argmax(axis=1))

# ATT: y_true must be the true indexes for y_pred 2D matrix
def compute_acc_acc5_f1_prec_rec(y_true, y_pred, print_metrics=False, print_pfx=''):
    if y_pred.ndim == 1:
        proc_y_pred = y_pred
    else:
        proc_y_pred = y_pred.argmax(axis=1)
    
    acc = accuracy(y_true, proc_y_pred)
    acc_top5 = accuracy_top_k(y_true, y_pred, K=5)
    bal_acc = balanced_accuracy(y_true, proc_y_pred)
    _f1_macro = f1_macro(y_true, proc_y_pred)
    _prec_macro = precision_macro(y_true, proc_y_pred)
    _rec_macro = recall_macro(y_true, proc_y_pred)

    if print_metrics:
        pfx = '' if print_pfx == '' else print_pfx + '\t\t'
        print(pfx + 'acc: %.6f\tacc_top5: %.6f\tf1_macro: %.6f\tprec_macro: %.6f\trec_macro: %.6f'
              % (acc, acc_top5, _f1_macro, _prec_macro, _rec_macro))
    
    return acc, acc_top5, _f1_macro, _prec_macro, _rec_macro, bal_acc

# ------------------------------------------------------------------------------
# From Movelets ML

def f1(y_true, y_pred):
#     from ..main import importer
    
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

# by Tarlis --
def calculateAccTop5(classifier, X_test, y_test, K ):
    K = K if len(y_test) > K else len(y_test)
    
    y_test_pred = classifier.predict_proba(X_test)
    order=np.argsort(y_test_pred, axis=1)
    n=classifier.classes_[order[:, -K:]]
    soma = 0;
    for i in range(0,len(y_test)) :
        if ( y_test[i] in n[i,:] ) :
            soma = soma + 1
    accTopK = soma / len(y_test)
    
    return accTopK

# by Tarlis --
def classification_report_csv(report, reportfile, classifier):
    report_data = []
    lines = report.split('\n')   
    for line in lines[2:(len(lines)-3)]:
        row_data = line.split()
        row = {}  
        
        if row_data == []:
            break
            
        row["class"] = row_data[0]
        row["classifier"] = classifier
        row["precision"] = float(row_data[1])
        row["recall"] = float(row_data[2])
        row["f1_score"] = float(row_data[3])
        row["support"] = float(row_data[4])

        report_data.append(row)
#     import pandas as pd
    dataframe = pd.DataFrame.from_dict(report_data)
    dataframe.to_csv(reportfile, index = False)
    return dataframe

def classification_report_dict2csv(report, reportfile, classifier): 
    dataframe = classification_report_dict2df(report, classifier)
    dataframe.to_csv(reportfile, index = False)
    return dataframe

def classification_report_dict2df(report, classifier):    
    report_data = []  
    for k, v in report.items():
        if k in ['accuracy', 'macro avg', 'weighted avg']:
            continue

        row = {}
        row["class"] = k
        row["classifier"] = classifier
        row["precision"] = float(v['precision'])
        row["recall"] = float(v['recall'])
        row["f1_score"] = float(v['f1-score'])
        row["support"] = float(v['support'])

        report_data.append(row)

    dataframe = pd.DataFrame.from_dict(report_data)
#    dataframe.to_csv(reportfile, index = False)
    return dataframe

# ------------------------------------------------------------------------------
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

        self._df = pd.concat([self._df, pd.DataFrame({
                                    'method': method,
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
                                    'test_acc_up': 1 if test_acc > test_max_acc else 0}, index=[len(self._df)])])
#        self._df = self._df.append({'method': method,
#                                    'epoch': epoch,
#                                    'dataset': dataset,
#                                    'timestamp': timestamp,
#                                    'train_loss': train_loss,
#                                    'train_acc': train_acc,
#                                    'train_acc_top5': train_acc_top5,
#                                    'train_f1_macro': train_f1_macro,
#                                    'train_prec_macro': train_prec_macro,
#                                    'train_rec_macro': train_rec_macro,
#                                    'train_acc_up': 1 if train_acc > train_max_acc else 0,
#                                    'test_loss': test_loss,
#                                    'test_acc': test_acc,
#                                    'test_acc_top5': test_acc_top5,
#                                    'test_f1_macro': test_f1_macro,
#                                    'test_prec_macro': test_prec_macro,
#                                    'test_rec_macro': test_rec_macro,
#                                    'test_acc_up': 1 if test_acc > test_max_acc else 0},
#                                   ignore_index=True)

    def save(self, file):
        self._df.to_csv(file, index=False)

    def load(self, file):
        if os.path.isfile(file):
            self._df = pd.read_csv(file)
        else:
            print("WARNING: File '" + file + "' not found!")

        return self

    
############################
# other evaluation metrics #
############################
def mape(true, pred, sample_weight=None):
    """
    it is very, very slow when the shapes are different and the dataset is very large (e.g. > 1,000,000 rows).
    """
    if true.shape != pred.shape:
        true = true.reshape(pred.shape)
    return np.average( np.abs( (true - pred) / true ), weights=sample_weight)

def mape_xgb(true, pred, sample_weight=None):
    return 'error', mape(true, pred.get_label(), sample_weight)

def mse(true, pred, sample_weight=None):
    if true.shape[0] == 0:
        return None
    return mean_squared_error(true, pred, sample_weight)

def mse_xgb(true, pred, sample_weight=None):
    return 'error', mse(true, pred.get_label(), sample_weight)

def regularity(sequence):
    return len(sequence) / len(set(sequence))    

def stationarity(sequence):
    rle = [(value, sum(1 for i in g)) for value, g in groupby(sequence)]
    return np.mean([length for _, length in rle])