# -*- coding: utf-8 -*-
'''
MAT-Tools: Python Framework for Multiple Aspect Trajectory Data Mining

The present package offers a tool, to support the user in the task of data analysis of multiple aspect trajectories. It integrates into a unique framework for multiple aspects trajectories and in general for multidimensional sequence data mining methods.
Copyright (C) 2022, MIT license (this portion of code is subject to licensing from source project distribution)

Created on Dec, 2021
Copyright (C) 2022, License GPL Version 3 or superior (see LICENSE file)

Authors:
    - Tarlis Portela
    - Francisco Vicenzi (adapted)
'''
# --------------------------------------------------------------------------------
import os 
import numpy as np
import pandas as pd
from numpy import argmax
#sys.path.insert(0, os.path.abspath('.')) # TODO fix imports

# --------------------------------------------------------------------------------
from tensorflow import random
from sklearn.preprocessing import scale, OneHotEncoder
from matdata.preprocess import trainTestSplit
  
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, History

from matclassification.methods._lib.metrics import compute_acc_acc5_f1_prec_rec
from sklearn.metrics import classification_report
#from matclassification.methods._lib.pymove.models import metrics
# --------------------------------------------------------------------------------
from matclassification.methods._lib.logger import Logger
from matclassification.methods._lib.metrics import MetricsLogger
from matclassification.methods.mat.MARC import EpochLogger
# --------------------------------------------------------------------------------
from matclassification.methods.core import HSClassifier

from matclassification.methods.feature.feature_extraction.pois import pois

class POIS(HSClassifier):
    """
    POIS: Point of Interest Sequence Feature Exctractor and Classifier.

    This class implements a trajectory classifier based on the POI-F/POIS approach, which considers the frequency of visits to 
    Points of Interest (POIs). It has been extended to concatenate sequences of POIs, which allows classification based on 
    patterns in POI sequences. 

    POI Frequency types:
    (i) poi: POI frequency
    (ii) npoi: Normalized POI frequency
    (iii) wnpoi: Weighted Normalized POI frequency

    Parameters:
    -----------
    method : str, optional (default='npoi')
        The method used to compute POI frequencies. Options include 'poi', 'npoi', and 'wnpoi'.
    
    sequences : list of int, optional (default=[1, 2, 3])
        Defines the length of the sequences of POIs used for classification.
    
    features : list, optional
        Specifies which features from the dataset to use. If None, choose the feature with higher variance.
    
    n_jobs : int, optional (default=-1)
        The number of parallel jobs to run for computation.
    
    verbose : bool, optional (default=True)
        Controls verbosity of logging during model training.
    
    random_state : int, optional (default=42)
        Seed used by the random number generator to ensure reproducibility.

    filterwarnings : str, optional (default='ignore')
        Controls the filter for output warnings.

    Methods:
    --------
    xy(train, test, tid_col, class_col, geo_precision, validate, res_path):
        Prepares the data for training and testing by computing the POI sequences and extracting features for classification.
    
    prepare_input(train, test, tid_col, class_col, geo_precision, validate, res_path):
        Prepares and splits the input data into training, validation, and testing sets. Adds the necessary configuration details.
    
    create():
        Initializes a neural network model with two layers: a hidden layer with 100 units and a softmax output layer for classification.

    fit(X_train, y_train, X_val, y_val, save_results, res_path):
        Trains the model using the training data. Optionally saves the results.

    predict(X_test, y_test):
        Predicts the labels for the test data and returns the classification report.
    """
    
    def __init__(self, 
                 method='npoi',
                 sequences=[1,2,3], 
                 features=None, 
#                 dataset='specific', #DEPRECATED
                 
                 n_jobs=-1,
                 verbose=True,
                 random_state=42,
                 filterwarnings='ignore'):
        
        super().__init__(method.upper(), n_jobs=n_jobs, verbose=verbose, random_state=random_state, filterwarnings=filterwarnings)
        
        self.add_config(method=method,
                        sequences=sequences, 
                        features=features) 
#                        dataset=dataset)
        
        np.random.seed(seed=random_state)
        random.set_seed(random_state)
        
    def xy(self,
           train, test,
           tid_col='tid', 
           class_col='label',
#          space_geohash=False, # For future implementation
           geo_precision=8,
           validate=False, # For future implementation
           res_path=None): 
        
        def check_label_sort(df):
            labels = list(df[class_col])
            indexes = [index for index, _ in enumerate(labels) if labels[index] != labels[index-1]]
            final = list(map(lambda i: labels[i], indexes))
            print(len(final), len(set(final)), len(final) != len(set(final)))
            return len(final) == len(set(final))
        
        assert check_label_sort(train) and check_label_sort(test), "This method requires input data to be ordered by labels."
        
        if res_path:
            save_results=True
        else:
            save_results=False
        
        sequences = self.config['sequences']
        features = self.config['features']
        method = self.config['method']
#        dataset = self.config['dataset'] # DEPRECATED
        
        random_num = self.config['random_state']
        
        X_train, X_test, y_train, y_test, _ = pois(train, test, 
                                                   sequences, features, method, 
                                                   res_path, save_results, tid_col, class_col, 
                                                   verbose=self.isverbose)
        
        # return: (num_features, num_classes, labels, X, y, one_hot_y)
        return prepareData(X_train, X_test, y_train, y_test,
                             validate=validate,random_state=random_num)
    
    def prepare_input(self,
                      train, test,
                      tid_col='tid', 
                      class_col='label',
#                      space_geohash=False, # For future implementation
                      geo_precision=8,
                      validate=False, # For future implementation
                      res_path=None): 
        
        (num_features, num_classes, labels, X, y, one_hot_y) = self.xy(train, test, tid_col, class_col, geo_precision, validate, res_path)
    
        self.add_config(num_features=num_features,
                        num_classes=num_classes, 
                        labels=labels)

        self.le = one_hot_y
        
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
        
        return X, y, self.config['num_features'], self.config['num_classes']
        
    
    def create(self):
        
        num_classes = self.config['num_classes']
        num_features = self.config['num_features']
        
        keep_prob = 0.5

        HIDDEN_UNITS = 100
        LEARNING_RATE = 0.001
#        EARLY_STOPPING_PATIENCE = 30
        
        model = Sequential()
        model.add(Dense(units=HIDDEN_UNITS,
                        input_dim=(num_features),
                        kernel_initializer='uniform',
                        kernel_regularizer=l2(0.02)))
        model.add(Dropout(keep_prob))
        model.add(Dense(units=num_classes,
                        kernel_initializer='uniform',
                        activation='softmax'))

        if num_classes < 5:
            my_metrics = ['acc']
        else:
            my_metrics = ['acc', 'top_k_categorical_accuracy']
        
        opt = Adam(lr=LEARNING_RATE)
        model.compile(optimizer=opt,
                      loss='categorical_crossentropy',
                      metrics=my_metrics)
        
        return model

        
    def fit(self, 
            X_train, 
            y_train, 
            X_val,
            y_val, 
            save_results=False,
            res_path='.'):
        
        # Check if config is done:
        if 'num_features' not in self.config.keys():
            labels = list(y_train)
            num_features = len(list(X_train.columns))
            num_classes = len(set(y_train))
            self.add_config(num_features=num_features,
                            num_classes=num_classes, 
                            labels=labels)
        
        EPOCHS = 250
        BASELINE_METRIC = 'acc'
        BASELINE_VALUE = 0.5
        BATCH_SIZE = 64
        EARLY_STOPPING_PATIENCE = 30
        
        METRICS_FILE = None
        if save_results:
            METRICS_FILE = self.name+'-results.csv'
            METRICS_FILE = os.path.join(res_path, METRICS_FILE)
        
        self.model = self.create()
        
        hist = History()
        
        history = self.model.fit(x=X_train,
                  y=y_train,
                  validation_data=(X_val, y_val),
                  batch_size=BATCH_SIZE,
                  shuffle=True,
                  epochs=EPOCHS,
                  verbose=0,
                  callbacks=[EpochLogger(X_train, y_train, X_val, y_val, 
                                          metric=BASELINE_METRIC,
                                          baseline=BASELINE_VALUE,
                                          patience=EARLY_STOPPING_PATIENCE,
                                          metrics_file=METRICS_FILE), hist])
        
        self.report = pd.DataFrame(history.history)
        return self.report
    
    def predict(self,                 
                X_test,
                y_test):
        
        y_pred = self.model.predict(X_test)#, y_test)
        self._summary = self.score(argmax(y_test, axis = 1), y_pred)
        
        print('NOW:', y_test.ndim, y_pred.ndim)
        
        self.y_test_true = y_test
        self.y_test_pred = y_pred
        
        if self.le:
            self.y_test_true = self.le.inverse_transform(self.y_test_true).reshape(1, -1)[0]
            self.y_test_pred = self.le.inverse_transform(self.y_test_pred).reshape(1, -1)[0]
        
        return self._summary, y_pred


###############################################################################    
def prepareData(x_train, x_test, y_train, y_test, validate=False, random_state=42):
    """
    Prepares the dataset for training, testing, and optional validation (#TODO) for POIS.

    Parameters:
    -----------
    x_train : pandas.DataFrame or numpy.ndarray
        Feature set for the training data.
        
    x_test : pandas.DataFrame or numpy.ndarray
        Feature set for the test data.
        
    y_train : pandas.Series or numpy.ndarray
        Labels for the training data.
        
    y_test : pandas.Series or numpy.ndarray
        Labels for the test data.
        
    validate : bool, optional (default=False)
        If True, splits the training data into training and validation sets. 
        Validation handling is currently not implemented.
        
    random_state : int, optional (default=42)
        Random seed used for reproducibility when splitting data.
        
    Returns:
    --------
    num_features : int
        The number of features in the dataset.
        
    num_classes : int
        The number of unique classes in the target labels.
        
    labels : numpy.ndarray
        An array of the unique class labels.
        
    X : list
        A list containing feature sets. If `validate` is False, returns 
        [X_train, X_test]. Otherwise, returns [X_train, X_val, X_test].
        
    y : list
        A list containing one-hot encoded target sets. If `validate` is False, returns 
        [y_train, y_test]. Otherwise, returns [y_train, y_val, y_test].
        
    one_hot_encoder : sklearn.preprocessing.OneHotEncoder
        The fitted OneHotEncoder object used to encode the target labels.
    """
    
    if validate:
        raise NotImplementedError('POIS method prepareData(validate=True) is not implemented.')
    
    labels = list(y_test)

    num_features = len(list(x_train))
    num_classes = len(set(y_train))
    
    if validate:
        df_xtrain = x_train.copy()
        df_ytrain = pd.DataFrame(y_train)
        df_xtrain['tid'] = df_xtrain.index
        df_ytrain['tid'] = df_ytrain.index
        
        return df_xtrain, df_ytrain, df_xtrain, df_ytrain

        df_xtrain, df_xval = trainTestSplit(df_xtrain, train_size=0.75,
                                             random_num=random_state, outformats=[])
        df_ytrain, df_yval = trainTestSplit(df_ytrain, train_size=0.75,
                                             random_num=random_state, outformats=[])

        df_xtrain.drop(columns=['tid'], inplace=True)
        df_xval.drop(columns=['tid'], inplace=True)
        df_ytrain.drop(columns=['tid'], inplace=True)
        df_yval.drop(columns=['tid'], inplace=True)

        data = [(df_xtrain, df_xval, x_test), (df_ytrain.T.values[0], df_yval.T.values[0], y_test)]
    else:
        data = [(x_train, x_test), (y_train, y_test)]
        
    
    X = []
    y = []
    
    one_hot_y = OneHotEncoder()
    one_hot_y.fit(pd.DataFrame(y_train))
#    one_hot_y.fit(y_train.loc[:, [class_col]])
    for df in data[1]:       
#        df = df.reshape(-1, 1)
        df = one_hot_y.transform(pd.DataFrame(df)).toarray()
#        df = one_hot_y.transform(df.loc[:, [class_col]]).toarray()
        y.append(df)
        
    for df in data[0]:
        df = scale(df, axis=0)
        X.append(df)
    
    return (num_features, num_classes, labels, X, y, one_hot_y)

# --------------------------------------------------------------------------------------------------------  
def loadData(dir_path):
    """
    Loads training and testing datasets from CSV files.

    Parameters:
    -----------
    dir_path : str
        The directory path (without file extension) from which to load the datasets. 
        It expects the following CSV files:
        - '{dir_path}-x_train.csv': Features for training data.
        - '{dir_path}-y_train.csv': Labels for training data.
        - '{dir_path}-x_test.csv': Features for testing data.
        - '{dir_path}-y_test.csv': Labels for testing data.

    Returns:
    --------
    x_train : pandas.DataFrame
        A DataFrame containing the features for the training dataset.
        
    x_test : pandas.DataFrame
        A DataFrame containing the features for the testing dataset.
        
    y_train : numpy.ndarray
        A numpy array containing the labels for the training dataset.
        
    y_test : numpy.ndarray
        A numpy array containing the labels for the testing dataset.
    """
    
    x_train = pd.read_csv(dir_path+'-x_train.csv')#, header=None)
    y_train = pd.read_csv(dir_path+'-y_train.csv').iloc[:,-1].values#[:,-1].values

    x_test = pd.read_csv(dir_path+'-x_test.csv')#, header=None)
    y_test = pd.read_csv(dir_path+'-y_test.csv').iloc[:,-1].values
    
    return x_train, x_test, y_train, y_test

