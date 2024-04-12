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
import os 
import numpy as np
import pandas as pd
from numpy import argmax
from datetime import datetime

from tqdm.auto import tqdm
# --------------------------------------------------------------------------------
from tensorflow import random
from matdata.preprocess import trainAndTestSplit
from matanalysis.methods._lib.datahandler import prepareTrajectories
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras.utils import to_categorical

from sklearn.metrics import classification_report
#from matanalysis.methods._lib.pymove.models import metrics
#from matanalysis.methods._lib.metrics import compute_acc_acc5_f1_prec_rec
#from matanalysis.methods._lib.metrics import classification_report_csv, classification_report_dict2csv
from matanalysis.methods._lib.metrics import *
from matanalysis.methods._lib.pymove.models import metrics
# --------------------------------------------------------------------------------
import warnings
# --------------------------------------------------------------------------------
from abc import ABC, abstractmethod
# TODO implement rounds

# Simple Abstract Classifier Model
class AbstractClassifier(ABC):
    
    def __init__(self, 
                 name='NN',
                 n_jobs=-1,
                 verbose=0,
                 random_state=42,
                 filterwarnings='ignore'):
        
        self.name = name
        self.y_test_pred = None
        self.model = None
        self.le = None
        
        self.isverbose = verbose >= 0
        
        self.save_results = False # TODO
        self.validate = False
        topK = 5
        
        self.config = dict()
        self.add_config(topK=topK,
                        n_jobs=n_jobs,
                        verbose=verbose,
                        random_state=random_state)
        
        
        if filterwarnings:
            warnings.filterwarnings(filterwarnings)
        
        if self.isverbose:
            print('\n['+self.name+':] Building model')
        self.start_time = datetime.now()
        
        #assert (eval_metric == 'merror') | (eval_metric == 'mlogloss'), "ERR: invalid loss, set loss as mlogloss or merror" 

        #print('[MODEL:] Starting model training, {} iterations'.format(total)
    
    def add_config(self, **kwargs):
        self.config.update(kwargs)
    
    def duration(self):
        return (datetime.now()-self.start_time).total_seconds() * 1000
    
    def message(self, pbar, text):
        if isinstance(pbar, list):
            print(text)
        else:
            pbar.set_postfix_str(text)

    @abstractmethod
    def create(self):
        
        # **** Method to overrite ****
        print('\n['+self.name+':] Warning! you must overwrite the create() method.')
        self.model = None
        
        return self.model

##    @abstractmethod
#    def fit(self, 
#            X_train, 
#            y_train, 
#            X_val,
#            y_val):
#        
#        # **** Method to overrite ****
##        print('\n['+self.name+':] Warning! you must overwrite the fit() method.')
#        self.model = self.create()
#        self.model.fit(X_train, y_train)
#        
#        y_pred = self.model.predict(X_val, y_val)
#        
#        self.report = self.score(X_val, y_val, y_pred)
#        
#        return self.report
    def fit(self, 
            X_train, 
            y_train, 
            X_val,
            y_val):
        
        self.model = self.create()
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_val)
        
        self.report = self.score(X_val, y_val, y_pred)
        
        return self.report
        
#    def fit(self, 
#            X_train, 
#            y_train, 
#            X_val,
#            y_val, 
#            verbose=True):
#        
#        eval_set = [(X_train, y_train), (X_val, y_val)]
#        
#        history = self.model.fit(X_train, y_train, 
#                      eval_set=eval_set,
#                      verbose=verbose)
#    
#        self.report = pd.DataFrame(history.history)
#        return self.report
   
#    def predict_proba(self,                 
#                X_test,
#                y_test):
#        return self.predict(X_test, y_test)
    
    def predict(self,                 
                X_test,
                y_test):
        
        y_pred = self.model.predict(X_test, y_test)
        
        self.y_test_true = argmax(y_test, axis = 1)
        self.y_test_pred = argmax(y_pred , axis = 1)
        
        if self.le:
            self.y_test_true = self.le.inverse_transform(self.y_test_true)
            self.y_test_pred = self.le.inverse_transform(self.y_test_pred)

        self._summary = self.score(X_test, y_test, y_pred)
            
        return self._summary, y_pred   
    
    def score(self, X_test, y_test, y_pred):
        acc, acc_top5, _f1_macro, _prec_macro, _rec_macro, bal_acc = compute_acc_acc5_f1_prec_rec(y_test, np.array(y_pred), print_metrics=False)
        
        dic_model = {
            'acc': acc,
            'acc_top_K5': acc_top5,
            'balanced_accuracy': bal_acc,
            'precision_macro': _f1_macro,
            'recall_macro': _prec_macro,
            'f1_macro': _rec_macro,
        } 
        
        return pd.DataFrame(dic_model, index=[0])

    
#    def train(self):
#        
#        X_train = self.X_train
#        y_train = self.y_train
#        
#        if self.validate:
#            X_val = self.X_val
#            y_val = self.y_val
#        else:
#            X_val = self.X_test
#            y_val = self.y_test   
#            
#        return self.fit(X_train, y_train, X_val, y_val)

#    def test(self):            
#        X_test = self.X_test
#        y_test = self.y_test
#        
#        return self.predict(X_test,y_test)

#    def test(self,
#             rounds=1,
#             dir_evaluation='.'): 
#        
#        if rounds > 1:
#            self.test_rounds(rounds, dir_evaluation)
#        else:
#            # Do test on trained model:
#            X_test = self.X_test
#            y_test = self.y_test
#
#            self.test_report, y_test_pred = self.predict(X_test,y_test)
#
#            if self.isverbose:
#                print('['+self.name+':] Processing time: {} milliseconds. Done.'.format(self.duration()))
#
#            return self.test_report, y_test_pred

    def test(self,
             rounds=1,
             dir_evaluation='.'):
        
        X_train = self.X_train
        y_train = self.y_train
        
        if self.validate:
            X_val = self.X_val
            y_val = self.y_val
        else:
            X_val = self.X_test
            y_val = self.y_test  
            
        X_test = self.X_test
        y_test = self.y_test
        
        filename = os.path.join(dir_evaluation, 'eval_'+self.name.lower()+'.csv')
        
        if os.path.exists(filename):
            if self.isverbose:
                print('['+self.name+':] Model previoulsy built.')
            # TODO read
            #return self.read_report(filename, prefix='eval_')
        else:
            if self.isverbose:
                print('['+self.name+':] Creating a model to test set')
            
                pbar = tqdm(range(rounds), desc="Model Testing")
            else:
                pbar = list(range(rounds))
                
            random_state = self.config['random_state']
            
            evaluate_report = []
            for e in pbar:
                re = (random_state+e)
                self.config['random_state'] = re
                
                self.message(pbar, 'Round {} of {} (random {})'.format(e, rounds, re))
                
                self.model = self.create()
                
                self.fit(X_train, y_train, X_val, y_val)
                
                eval_report, y_pred = self.predict(X_test, y_test)
                evaluate_report.append(eval_report)
                        
            self.config['random_state'] = random_state
            self.test_report = pd.concat(evaluate_report)
            self.test_report.reset_index(drop=True, inplace=True)
            
            if self.isverbose:
                print('['+self.name+':] Processing time: {} milliseconds. Done.'.format(self.duration()))

            return self.test_report, y_pred
    
#    def label_decode(self):
#        if self.le:
#            self.y_test_true = self.le.inverse_transform(self.y_test_true)
#            self.y_test_pred = self.le.inverse_transform(self.y_test_pred)
#            
#    def prediction(self,                 
#                X_test,
#                y_test):
#        if not self.y_test_pred:
#            self.predict(X_test,y_test)
#            self.label_decode()
#                
#        self.predicted = True
#        return pd.DataFrame(self.y_test_true,self.y_test_pred)
#
#    def prediction(self):
#        if self.y_test_pred:
#            return pd.DataFrame(self.y_test_true,self.y_test_pred)
#        else:
#            return None

    
#    def container(self):
#        return ModelContainer(self.classifier, self.y_test_true, self.x_test, cls_history=history.history, approach='MLP', le=le)

#    def report(self):
#        return classification_report(self.y_test_true, self.y_test_pred, output_dict=True)  

    
    def summary(self):
        return pd.DataFrame(self.test_report.mean()).T
    
#    def summary(self):
#        return self.result
#        tail = self.result.tail(1)
#        summ = {
#            'acc':               tail['acc'].values,
#            'acc_top_K5':        tail['acc_top_K5'].values,
#            'balanced_accuracy': tail['balanced_accuracy'].values,
#            'precision_macro':   tail['precision_macro'].values,
#            'recall_macro':      tail['recall_macro'].values,
#            'f1_macro':          tail['f1-macro'].values,
#            'loss':              None,
#        }
#        
#        return pd.DataFrame(summ)
        
    def save(self, dir_path='.', modelfolder='model'):
        if not os.path.exists(os.path.join(dir_path, modelfolder)):
            os.makedirs(os.path.join(dir_path, modelfolder))
        self.model.save(os.path.join(dir_path, modelfolder, 'model_'+self.name.lower()+'.h5'))

        prediction = self.prediction()
        prediction.to_csv(os.path.join(dir_path, modelfolder, 'model_'+self.name.lower()+'_prediction.csv'), header = True) 
        
        report = self.report()
        classification_report_dict2csv(report, os.path.join(dir_path, modelfolder, 'model_'+self.name.lower()+'_report.csv'), self.approach)
        self.report.to_csv(os.path.join(dir_path, modelfolder, 'model_'+self.name.lower()+'_history.csv'))

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
        
    # Mabe comment?
#    def score(self, X_test, y_test, y_pred):
#        acc = self.model.score(X_test,y_test)
#        acc_top5 = calculateAccTop5(self.model, X_test, y_test, self.config['topK'])
#        bal_acc = balanced_accuracy(y_test, y_pred)
#        _f1_macro = f1_score(y_test, y_pred, average='macro')
#        _prec_macro = precision_score(y_test, y_pred, average='macro', zero_division=1)
#        _rec_macro = recall_score(y_test, y_pred, average='macro')
#        
#        dic_model = {
#            'acc': acc,
#            'acc_top_K5': acc_top5,
#            'balanced_accuracy': bal_acc,
#            'precision_macro': _f1_macro,
#            'recall_macro': _prec_macro,
#            'f1_macro': _rec_macro,
#        } 
#        
#        return pd.DataFrame(dic_model, index=[0])
    
    def prepare_input(self,
                      train, test,
                      
                      tid_col='tid', class_col='label',
                      geo_precision=30, # TODO for future implementation
                      validate = True):
        
        #Preparing the input of movelets:
        #nattr = len(train.iloc[1,:])
        #nattr = nattr-1 if tid_col in train.columns else nattr
        #print("Number of attributes: " + str(nattr))
        
        data = []
        if validate:
            df_train = train.copy()
            if tid_col not in df_train.columns:
                df_train[tid_col] = df_train.index
            
            df_train, df_val = trainAndTestSplit(df_train, train_size=0.75, 
                                                 tid_col=tid_col, class_col=class_col, 
                                                 random_num=self.config['random_state'], outformats=[])
            
#            df_train.drop(columns=[tid_col], inplace=True)
#            df_val.drop(columns=[tid_col], inplace=True)
#            test.drop(columns=[tid_col], inplace=True)
            
            data = [df_train, df_val, test]
        else:
            data = [train, test]
        
        for df in data:
            if tid_col in df.columns:
                df.drop(columns=[tid_col], inplace=True)
                
        
        self.config['num_classes'] = len(train[class_col].unique())
        
        nattr = len(data[0].iloc[1,:])
        #print("Number of attributes: " + str(nattr))
        self.config['num_features'] = nattr
        

        # Scaling y and transforming to keras format
        self.le = LabelEncoder()
        self.le.fit(train[class_col])
        
        # For Scaling data
        min_max_scaler = MinMaxScaler()
            
        X_set = []
        y_set = []
        # Separating attribute data (X) than class attribute (y)
        for dataset in data:            
            X = dataset.iloc[:, 0:(nattr-1)].values
            y = dataset.iloc[:, (nattr-1)].values
            
            # Replace distance 0 for presence 1
            # and distance 2 to non presence 0
            X[X == 0] = 1
            X[X == 2] = 0
            X = min_max_scaler.fit_transform(X)
            
            y = self.le.transform(y) 
            y = to_categorical(y)
            
            X_set.append(X)
            y_set.append(y)

        
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
            
        return X, y, self.config['num_features'], self.config['num_classes']
    
#    def predict(self,                 
#                X_test,
#                y_test):
#        
#        y_pred = self.model.predict(X_test, y_test)
#        
##        self.y_test_true = y_test
##        self.y_test_pred = y_pred
#        self.y_test_true = argmax(y_test, axis = 1)
#        self.y_test_pred = argmax(y_pred , axis = 1)
#        
#        if self.le:
#            self.y_test_true = self.le.inverse_transform(self.y_test_true)
#            self.y_test_pred = self.le.inverse_transform(self.y_test_pred)
#            
#        self._summary = self.score(X_test, y_test, y_pred)
#        
##        if self.isverbose:
##            print('['+self.name+':] Processing time: {} milliseconds. Done.'.format(self.duration()))
#            
#        return self._summary, y_pred
    
    def predict(self,                 
                X_test,
                y_test):
        
#        y_pred = self.model.predict_proba(X_test)
        y_pred = self.model.predict(X_test)
        
        self.y_test_true = argmax(y_test, axis = 1) #y_test
        self.y_test_pred = argmax(y_pred , axis = 1)
        
        if self.le:
            self.y_test_true = self.le.inverse_transform(self.y_test_true)
            self.y_test_pred = self.le.inverse_transform(self.y_test_pred)
            
        self._summary = self.score(X_test, y_test, y_pred) #self.y_test_pred)
        
        return self._summary, y_pred 
    
    ## Overwrite train method to do Hiperparameter Optimization:
    def train(self, dir_validation='.'):
        
        # This implementation, trains only one model 
        # (but, you may overwrite the method following this structure or HPSClassifier.train())
        
        X_train = self.X_train
        y_train = self.y_train
        
        if self.validate:
            X_val = self.X_val
            y_val = self.y_val
        else:
            X_val = self.X_test
            y_val = self.y_test            
        
        if self.isverbose:
            print('['+self.name+':] Training model')
                
        data = []
        ## This part you may want to run for each configuration (as a progress bar):
        #for config in pbar:
        filename = os.path.join(dir_validation, 'val_'+self.name.lower()+'.csv')
            
        if os.path.exists(filename):
            print('Skip ---> {}'.format(filename)) #TODO LOAD results
        else:
            self.model = self.create()
            self.fit(X_train, y_train, X_val, y_val)

            validation_report, y_pred = self.predict(X_val, y_val)

            if self.save_results:
                validation_report.to_csv(filename, index=False)

            data.append( validation_report )

#                self.model.free()
        
        self.report = pd.concat(data)
        self.report.reset_index(drop=True, inplace=True)

        self.report.sort_values('acc', ascending=False, inplace=True)
        
        return self.report
    
    # TODO test method
        

# Hiperparameter Optimization Classifier - For Movelet/Features input data
class MHPOClassifier(MClassifier):
    
    def __init__(self, 
                 name='NN',
                 save_results=False,
                 n_jobs=-1,
                 verbose=False,
                 random_state=42,
                 filterwarnings='ignore'):
        
        super().__init__(name=name, n_jobs=n_jobs, verbose=verbose, random_state=random_state, filterwarnings=filterwarnings)
        
        self.save_results = save_results
        
        np.random.seed(seed=random_state)
        random.set_seed(random_state)
    
    def score(self, X_test, y_test, y_pred):
        acc, acc_top5, _f1_macro, _prec_macro, _rec_macro, bal_acc = compute_acc_acc5_f1_prec_rec(y_test, np.array(y_pred), print_metrics=False)
        
        dic_model = {
            'acc': acc,
            'acc_top_K5': acc_top5,
            'balanced_accuracy': bal_acc,
            'precision_macro': _f1_macro,
            'recall_macro': _prec_macro,
            'f1_macro': _rec_macro,
        } 
        
        return pd.DataFrame(dic_model, index=[0])
    
    def create(self, config=None):
        
        # **** Method to overrite ****
        print('\n['+self.name+':] Warning! you must overwrite the create() method.')

        # Example structure:
        if hasattr(self, 'best_config'):
            self.model = None
        else:
            self.model = None
        
        return self.model
    
    
    ## Overwrite train method to do Hiperparameter Optimization:
    def train(self, dir_validation='.'):
        
        # This implementation, trains only one model 
        # (but, you may overwrite the method following this structure or HPSClassifier.train())
        
        X_train = self.X_train
        y_train = self.y_train
        
        if self.validate:
            X_val = self.X_val
            y_val = self.y_val
        else:
            X_val = self.X_test
            y_val = self.y_test            
        
        if self.isverbose:
            print('['+self.name+':] Training hiperparameter model')
        
        data = []
        
        ## This part you may want to run for each configuration (as a progress bar):
        #for config in pbar:
        filename = os.path.join(dir_validation, 'val_'+self.name.lower()+'.csv')
            
        if os.path.exists(filename):
            print('Skip ---> {}'.format(filename))
        else:
            self.model = self.create() # pass the config dict()
            self.fit(X_train, y_train, X_val, y_val) #, config)

            validation_report, y_pred = self.model.predict(X_val, y_val)

            if self.save_results:
                validation_report.to_csv(filename, index=False)

            data.append( validation_report )

#                self.model.free()
        
        self.report = pd.concat(data)
        self.report.reset_index(drop=True, inplace=True)

        self.report.sort_values('acc', ascending=False, inplace=True)
        
        return self.report
    
# --------------------------------------------------------------------------------
# Hiperparameter Optimization Classifier - For Trajectory input data
class HPOClassifier(AbstractClassifier):
    
    def __init__(self, 
                 name='NN',
                 save_results=False,
                 n_jobs=-1,
                 verbose=False,
                 random_state=42,
                 filterwarnings='ignore'):
        
        super().__init__(name=name, n_jobs=n_jobs, verbose=verbose, random_state=random_state, filterwarnings=filterwarnings)
        
        self.save_results = save_results
        
    def message(self, pbar, text):
        if self.isverbose:
            pbar.set_postfix_str(text)
                
    def read_report(self, filename, prefix=''):
        marksplit = '-'
        validation_report = pd.read_csv(filename)
        filename = filename.split(prefix+self.name.lower()+'-')[-1]
        filename = filename.split('.csv')[0]

        i = 0
        for y in filename.split(marksplit):
            validation_report['p'+str(i)] = y
            i+=1
        return validation_report
    
    def prepare_input(self,
                      train, test,
                      tid_col='tid', 
                      class_col='label',
                      space_geohash=False, # True: Geohash, False: indexgrid
                      geo_precision=30,    # Geohash: precision OR IndexGrid: meters
                      validate=True):
        
        # Load Data - Tarlis:
        X, y, features, num_classes, space, dic_parameters = prepareTrajectories(train.copy(), test.copy(),
                                                                                 tid_col=tid_col, 
                                                                                 class_col=class_col,
                                                                                 # space_geohash, True: Geohash, False: indexgrid
                                                                                 space_geohash=space_geohash, 
                                                                                 # Geohash: precision OR IndexGrid: meters
                                                                                 geo_precision=geo_precision,     
                                                                                 
                                                                                 features_encoding=True, 
                                                                                 y_one_hot_encodding=False,
                                                                                 split_test_validation=validate,
                                                                                 data_preparation=1,
                                                                                 
                                                                                 verbose=self.isverbose)
        self.config['features'] = features
        self.config['num_classes'] = num_classes
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
            
        return X, y, features, num_classes, space, dic_parameters
    
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
        
        return self.model.fit(X_train, 
                              y_train, 
                              X_val,
                              y_val)
    
    def predict(self,                 
                X_test,
                y_test):
        
#        self._summary, y_pred = self.model.predict(X_test, y_test)
        y_pred = self.model.model.predict_proba(X_test) 
        
#        print('Generating Classification Report')
        self._summary = metrics.compute_acc_acc5_f1_prec_rec(y_test, y_pred)
        
        self.y_test_true = y_test
        self.y_test_pred = y_pred
        
        if self.le:
            self.y_test_true = self.le.inverse_transform(self.y_test_true)
            self.y_test_pred = self.le.inverse_transform(self.y_test_pred)
            
        return self._summary, y_pred 
        
    def train(self, dir_validation='.'):
        
        X_train = self.X_train
        y_train = self.y_train
        
        if self.validate:
            X_val = self.X_val
            y_val = self.y_val
        else:
            X_val = self.X_test
            y_val = self.y_test            
        
        if self.isverbose:
            print('['+self.name+':] Training hiperparameter model')
        
        # Hiper-param data:
        data = []
        
        self.best_config = [-1, []]
        
        if self.isverbose:
            pbar = tqdm(self.grid, desc='['+self.name+':] Model Training')
        else:
            pbar = self.grid
        
        for config in pbar:
            
            # concat config:
            params = '-'.join([str(y) for y in config])
            filename = os.path.join(dir_validation, self.name.lower()+'-'+params+'.csv')
            
            if os.path.exists(filename):
                self.message(pbar, 'Skip ---> {}'.format(filename))
                data.append(self.read_report(filename))
                
            else:
                self.message(pbar, 'Trainning Config - '+params)
                
                self.model = self.create(config)
                self.fit(X_train, y_train, X_val, y_val, config)

                validation_report, y_pred = self.model.predict(X_val, y_val)

                if self.save_results:
                    validation_report.to_csv(filename, index=False)

                for index, (att, val) in enumerate(zip(['p'+str(y) for y in range(len(config))], config)):
                    validation_report[att] = [val]

                data.append( validation_report )
                
#                self.model.free()
                
            acc = validation_report.iloc[0]['acc']
            if acc > self.best_config[0]:
                self.best_config = [acc, config]
                self.best_model = self.model
            break # TODO: TEMP TESTING

        self.best_config = self.best_config[1]
        
        self.report = pd.concat(data)
        self.report.reset_index(drop=True, inplace=True)

        self.report.sort_values('acc', ascending=False, inplace=True)
        
        self.model = self.best_model
        
        return self.report
        
    def test(self,
             rounds=1,
             dir_evaluation='.'):
        
        X_train = self.X_train
        y_train = self.y_train
        
        if self.validate:
            X_val = self.X_val
            y_val = self.y_val
        else:
            X_val = self.X_test
            y_val = self.y_test  
            
        X_test = self.X_test
        y_test = self.y_test  
        
        params = '-'.join([str(y) for y in self.best_config])
        filename = os.path.join(dir_evaluation, 'eval_'+self.name.lower()+'-'+params+'.csv')
        
        if os.path.exists(filename):
            if self.isverbose:
                print('['+self.name+':] Model previoulsy built.')
            return self.read_report(filename, prefix='eval_')
        else:
            if self.isverbose:
                print('['+self.name+':] Creating a model to test set')
                print('['+self.name+':] Evaluation Config - '+params)
            
                pbar = tqdm(range(rounds), desc="Model Testing")
            else:
                pbar = list(range(rounds))
                
            random_state = self.config['random_state']
            
            evaluate_report = []
            for e in pbar:
                re = (random_state+e)
                self.config['random_state'] = re
                
                self.message(pbar, 'Round {} of {} (random {})'.format(e, rounds, re))
                
                self.model = self.create(self.best_config)
                
                self.fit(X_train, y_train, X_val, y_val, self.best_config)
                
#                eval_report, y_pred = self.model.predict(X_test, y_test)
                eval_report, y_pred = self.predict(X_test, y_test)
                evaluate_report.append(eval_report)
            
                        
            self.config['random_state'] = random_state
            
            self.test_report = pd.concat(evaluate_report)
            self.test_report.reset_index(drop=True, inplace=True)

#            self.test_report.sort_values('acc', ascending=False, inplace=True)
            
            if self.isverbose:
                print('['+self.name+':] Processing time: {} milliseconds. Done.'.format(self.duration()))

            return self.test_report, y_pred

    def summary(self):
#        tail = self.report.tail(1)
        tail = self.test_report.mean()
        summ = {
            'acc':               tail['acc'],
            'acc_top_K5':        tail['acc_top_K5'],
            'balanced_accuracy': tail['balanced_accuracy'],
            'precision_macro':   tail['precision_macro'],
            'recall_macro':      tail['recall_macro'],
            'f1_macro':          tail['f1-macro'],
            'loss':              None,
        }
        
        self._summary = pd.DataFrame(summ, index=[0])
        return self._summary
# --------------------------------------------------------------------------------->  