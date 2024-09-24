# -*- coding: utf-8 -*-
'''
MAT-classification: Analisys and Classification methods for Multiple Aspect Trajectory Data Mining

The present package offers a tool, to support the user in the task of data analysis of multiple aspect trajectories. It integrates into a unique framework for multiple aspects trajectories and in general for multidimensional sequence data mining methods.
Copyright (C) 2022, MIT license (this portion of code is subject to licensing from source project distribution)

Created on Dec, 2021
Copyright (C) 2022, License GPL Version 3 or superior (see LICENSE file)

@author: Tarlis Portela
'''
# --------------------------------------------------------------------------------
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from sklearn.utils import gen_even_slices

from matmodel.util.parsers import df2trajectory

from matsimilarity.methods.mat.MUITAS import *
#from matsimilarity.core.utils import similarity_matrix

# --------------------------------------------------------------------------------
from matclassification.methods.core import *

class SimilarityClassifier(THSClassifier):
    def __init__(self,
                 name,
                 
                 save_results=False,
                 n_jobs=-1,
                 verbose=False,
                 random_state=42,
                 filterwarnings='ignore'):
        
        super().__init__(name=name, save_results=save_results, n_jobs=n_jobs, verbose=verbose, random_state=random_state, filterwarnings=filterwarnings)
        
    def default_metric(self, dataset_descriptor):
        # Default similarity metric is MUITAS:
        muitas = MUITAS(dataset_descriptor)

        # Default Config:
        for feat in dataset_descriptor.attributes:
            muitas.add_feature([feat], 1)

        return muitas    
    
    def xy(self,
           train, test,
           tid_col='tid', 
           class_col='label',
           space_geohash=False, # True: Geohash, False: indexgrid
           geo_precision=30,    # Geohash: precision OR IndexGrid: meters
           validate=False,
           metric=None, 
           dataset_descriptor=None,
           inverse=True):
        
        train, dd = df2trajectory(train.copy())
        test, _ = df2trajectory(test.copy())
        
        if dataset_descriptor == None:
            dataset_descriptor = dd
            
        if metric == None:
            if self.isverbose:
                print('\n['+self.name+':] Default metric set to MUITAS.')
                
            self.metric = self.default_metric(dataset_descriptor)
        else:
            if self.isverbose:
                print('\n[{}:] Metric provided - {}.'.format(self.name, metric.__class__.__name__))
            self.metric = metric
        
        y_train = list(map(lambda t1: t1.label, train))
        y_test  = list(map(lambda t1: t1.label, test))
        
        self.le = LabelEncoder()
        self.le.fit(y_train)
        y = [
            self.le.transform(y_train),
            self.le.transform(y_test)
        ]
        
        X = list()
        if inverse: # Use inverse of similarity (distance metric):
            X.append( 1 - similarity_matrix(train, measure=self.metric, n_jobs=self.config['n_jobs']) )
            X.append( 1 - similarity_matrix(test, train,  measure=self.metric, n_jobs=self.config['n_jobs']) )
        else:
            X.append( similarity_matrix(train, measure=self.metric, n_jobs=self.config['n_jobs']) )
            X.append( similarity_matrix(test, train,  measure=self.metric, n_jobs=self.config['n_jobs']) )
        
        return X, y
    
    def prepare_input(self,
                      train, test,
                      tid_col='tid', 
                      class_col='label',
                      space_geohash=False, # True: Geohash, False: indexgrid
                      geo_precision=30,    # Geohash: precision OR IndexGrid: meters
                      validate=False,
                      metric=None, 
                      dataset_descriptor=None,
                      inverse=True):
        
        # Load Data - Tarlis:
        X, y = self.xy(train, test, tid_col, class_col, space_geohash, geo_precision, validate, metric, dataset_descriptor, inverse)
        
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
            
        return X, y
    
    def fit(self, 
            X_train, 
            y_train, 
            X_val=None,
            y_val=None,
            config=None):
        
        if not config:
            config = self.best_config            
        if self.model == None:
            self.model = self.create(config)
        
        return self.model.fit(X_train, y_train)
    
    def predict(self,                 
                X_test,
                y_test):
        
        y_pred_prob = self.model.predict_proba(X_test) 
        y_pred = argmax(y_pred_prob , axis = 1)

        self.y_test_true = y_test
        self.y_test_pred = y_pred
        
        if self.le:
            self.y_test_true = self.le.inverse_transform(self.y_test_true)
            self.y_test_pred = self.le.inverse_transform(self.y_test_pred)
        
        self._summary = self.score(y_test, y_pred_prob)
            
        return self._summary, y_pred_prob 
    
    
def similarity_matrix(A, B=None, measure=None, n_jobs=1):
    max_p = max(map(lambda t1: t1.size, A))
    
    def compute_slice(A, B, s):
        matrix = np.zeros(shape=(len(A), len(B), max_p, max_p, len(A[0].attributes)))

        for i in tqdm(range(s.start + 1, len(A)), desc='Computing similarity matrix'):
            for j in range(0, min(len(B), i - s.start)):
                matrix[i][j] = measure.similarity(A[i], B[j])
        return matrix

    upper = B is not None
    B = A if not B else B
    func = delayed(compute_slice)

    similarity = Parallel(n_jobs=n_jobs, verbose=0)(
        func(A, B[s], s) for s in gen_even_slices(len(B), n_jobs))
    similarity = np.hstack(similarity)

    if not upper:
        similarity += similarity.transpose() + np.identity(len(A))

    return similarity