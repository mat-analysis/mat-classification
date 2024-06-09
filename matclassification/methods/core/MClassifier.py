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
from matclassification.methods.core import *
# --------------------------------------------------------------------------------  

# Generic Movelet Classifier
class MClassifier(AbstractClassifier):
    
    def __init__(self, 
                 name='NN',
                 n_jobs=-1,
                 verbose=0,
                 random_state=42,
                 filterwarnings='ignore'):
        
        super().__init__(name=name, n_jobs=n_jobs, verbose=verbose, random_state=random_state, filterwarnings=filterwarnings)
        
    def xy(self,
           train, test,
           
           tid_col='tid', 
           class_col='label',
           validate = False,
           encode_labels=True):
        
        
        assert (len( set(test.columns).symmetric_difference(set(train.columns)) ) == 0), '['+self.name+':] ERROR. Divergence in train and test columns: ' + str(len(train.columns)) + ' train and ' + str(len(test.columns)) + ' test'
        
        data = []
        if validate:
            df_train = train.copy()
            if tid_col not in df_train.columns:
                df_train[tid_col] = df_train.index
            
            df_train, df_val = trainTestSplit(df_train, train_size=0.75, 
                                                 tid_col=tid_col, class_col=class_col, 
                                                 random_num=self.config['random_state'], outformats=[])
            
            data = [df_train, df_val, test]
        else:
            data = [train, test]
        
        for df in data:
            df.drop(columns=[tid_col], errors="ignore", inplace=True)
        
        num_classes = len(train[class_col].unique())
        num_features = len(data[0].iloc[1,:]) -1 # Minus label

        # Scaling y and transforming to keras format
        le = None
        if encode_labels:
            le = LabelEncoder()
            le.fit(train[class_col])
        
        # For Scaling data
        min_max_scaler = None
            
        X_set = []
        y_set = []

        for dataset in data:
            # Separating attribute data (X) than class attribute (y)
            X = dataset.iloc[:, 0:(num_features)].values
            y = dataset.iloc[:, (num_features)].values
            
            # Replace distance 0 for presence 1
            # and distance 2 to non presence 0
            X[X == 0] = 1
            X[X == 2] = 0
            
            # Scaling data
            if not min_max_scaler:
                min_max_scaler = MinMaxScaler()
                min_max_scaler.fit(X)
                
            X = min_max_scaler.transform(X)
            
            if encode_labels:
                y = le.transform(y)
                y = to_categorical(y)
            
            X_set.append(X)
            y_set.append(y)
        
        return num_classes, num_features, le, X_set, y_set
    
    def prepare_input(self,
                      train, test,
                      
                      tid_col='tid', 
                      class_col='label',
                      validate = False):
        
        num_classes, num_features, le, X_set, y_set = self.xy(train, test, tid_col, class_col, validate)
        
        self.add_config(num_classes=num_classes,
                        num_features=num_features)
        self.le = le
        
        if len(X_set) == 2:
            self.X_train = X_set[0] 
            self.X_test = X_set[1]
            self.y_train = y_set[0] 
            self.y_test = y_set[1]
            self.validate = False
        if len(X_set) > 2:
            self.X_train = X_set[0] 
            self.X_val = X_set[1]
            self.X_test = X_set[2]
            self.y_train = y_set[0] 
            self.y_val = y_set[1]
            self.y_test = y_set[2]
            self.validate = True
            
        return X_set, y_set, num_features, num_classes