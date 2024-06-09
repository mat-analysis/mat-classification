# -*- coding: utf-8 -*-
'''
MAT-Classification: Analisys and Classification methods for Multiple Aspect Trajectory Data Mining

The present package offers a tool, to support the user in the task of data analysis of multiple aspect trajectories. It integrates into a unique framework for multiple aspects trajectories and in general for multidimensional sequence data mining methods.
Copyright (C) 2022, MIT license (this portion of code is subject to licensing from source project distribution)

Created on Dec, 2021
Copyright (C) 2022, License GPL Version 3 or superior (see LICENSE file)

@author: Tarlis Portela
@author: Lucas May Petry (adapted)
'''
# --------------------------------------------------------------------------------
import os 
import numpy as np
import pandas as pd
from numpy import argmax
#sys.path.insert(0, os.path.abspath('.')) # TODO fix imports

# --------------------------------------------------------------------------------
from tensorflow import random
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from matclassification.methods._lib.geohash import bin_geohash
  
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout
from tensorflow.keras.initializers import he_uniform
from tensorflow.keras.regularizers import l1
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Add, Average, Concatenate, Embedding
from tensorflow.keras.callbacks import EarlyStopping

from matclassification.methods._lib.metrics import compute_acc_acc5_f1_prec_rec
from sklearn.metrics import classification_report
# --------------------------------------------------------------------------------
from matclassification.methods._lib.logger import Logger
from matclassification.methods._lib.metrics import MetricsLogger
# --------------------------------------------------------------------------------
from matclassification.methods.core import HSClassifier

class MARC(HSClassifier):
    
    def __init__(self, 
                 embedder_size=100, # [100, 200, 300] ?
                 merge_type='concatenate', # ['add', 'average', 'concatenate']
                 rnn_cell='lstm', # ['gru', 'lstm']
                 
                 # This are Default:
                 class_dropout = 0.5,
                 class_hidden_units = 100,
                 class_lrate = 0.001,
                 class_batch_size = 64,
                 class_epochs = 1000,
                 early_stopping_patience = 30,
                 baseline_metric = 'acc',
                 baseline_value = 0.5,
                 
                 n_jobs=-1,
                 verbose=True,
                 random_state=42,
                 filterwarnings='ignore'):
        
        super().__init__('MARC', n_jobs=n_jobs, verbose=verbose, random_state=random_state, filterwarnings=filterwarnings)
        
        VALID_MERGES = ['add', 'average', 'concatenate']
        VALID_CELLS = ['lstm', 'gru']
        
        assert merge_type in VALID_MERGES, "["+self.name+":] Merge type '" + merge_type + "' is not valid! Please choose 'add', 'average', or 'concatenate'."
        assert rnn_cell in VALID_CELLS, "["+self.name+":] RNN cell type '" + rnn_cell + "' is not valid! Please choose 'lstm' or 'gru'."
        
        np.random.seed(seed=random_state)
        random.set_seed(random_state)
        
        self.validate = False # TODO: Future impl
        
        self.add_config(embedder_size=embedder_size,
                        merge_type=merge_type,
                        rnn_cell=rnn_cell,
                        
                        class_dropout=class_dropout, 
                        class_hidden_units=class_hidden_units, 
                        class_lrate=class_lrate, 
                        class_batch_size=class_batch_size, 
                        class_epochs=class_epochs, 
                        early_stopping_patience=early_stopping_patience, 
                        baseline_metric=baseline_metric, 
                        baseline_value=baseline_value)
        
    def xy(self,
           train, test,
           tid_col='tid', 
           class_col='label',
#          space_geohash=True, # Always True for MARC
           geo_precision=8,
           validate=False): # validate - unused, for future implementation (TODO)

        (keys, vocab_size,
         num_classes,
         max_length,
         le,
         x_train, x_test,
         y_train, y_test) = prepareTrajectories(train.copy(), test.copy(),
                                             tid_col=tid_col,
                                             label_col=class_col,
                                             logger=Logger() if self.isverbose else None,
                                             geo_precision=geo_precision)
        
        x_train = [pad_sequences(f, max_length, padding='pre') for f in x_train]
        x_test = [pad_sequences(f, max_length, padding='pre') for f in x_test]
        
        return (keys, vocab_size, num_classes, max_length, le, 
                x_train, x_test, y_train, y_test)
    
    def prepare_input(self,
                      train, test,
                      tid_col='tid', 
                      class_col='label',
#                      space_geohash=True, # Always True for MARC
                      geo_precision=8,
                      validate=False): # validate - unused, for future implementation (TODO)
        
        (keys, vocab_size,
         num_classes,
         max_length,
         le,
         x_train, x_test,
         y_train, y_test) = self.xy(train, test, tid_col, class_col, geo_precision, validate)
        
        self.add_config(keys=keys, 
                        vocab_size=vocab_size,
                        num_classes=num_classes,
                        max_length=max_length)
        
        self.le = le
        
        self.X_train = x_train #[pad_sequences(f, max_length, padding='pre') for f in x_train]
        self.X_test = x_test #[pad_sequences(f, max_length, padding='pre') for f in x_test]
        self.y_train = y_train
        self.y_test = y_test
        
        return self.X_train, self.y_train, self.X_test, self.y_test
        
    
    def create(self):
        
        keys = self.config['keys']
        vocab_size = self.config['vocab_size']
        num_classes = self.config['num_classes']
        max_length = self.config['max_length']
        
        embedder_size = self.config['embedder_size']
        merge_type = self.config['merge_type']
        rnn_cell = self.config['rnn_cell']
        
        class_dropout = self.config['class_dropout']
        class_hidden_units = self.config['class_hidden_units']
        class_lrate = self.config['class_lrate']
        
#        early_stopping_patience = self.config['early_stopping_patience']
#        class_batch_size = self.config['class_batch_size']
#        class_epochs = self.config['class_epochs']
#        baseline_metric = self.config['baseline_metric']
#        baseline_value = self.config['baseline_value']
        
        inputs = []
        embeddings = []

        for idx, key in enumerate(keys):
            if key == 'lat_lon':
                i = Input(shape=(max_length, vocab_size[key]),
                          name='input_' + key)
                e = Dense(units=embedder_size,
                          kernel_initializer=he_uniform(seed=1),
                          name='emb_' + key)(i)
            else:
                i = Input(shape=(max_length,),
                          name='input_' + key)
                e = Embedding(vocab_size[key],
                              embedder_size,
                              input_length=max_length,
                              name='emb_' + key)(i)
            inputs.append(i)
            embeddings.append(e)

        if len(embeddings) == 1:
            hidden_input = embeddings[0]
        elif merge_type == 'add':
            hidden_input = Add()(embeddings)
        elif merge_type == 'average':
            hidden_input = Average()(embeddings)
        else:
            hidden_input = Concatenate(axis=2)(embeddings)

        hidden_dropout = Dropout(class_dropout)(hidden_input)

        if rnn_cell == 'lstm':
            rnn_cell = LSTM(units=class_hidden_units,
                            recurrent_regularizer=l1(0.02))(hidden_dropout)
        else:
            rnn_cell = GRU(units=class_hidden_units,
                           recurrent_regularizer=l1(0.02))(hidden_dropout)

        rnn_dropout = Dropout(class_dropout)(rnn_cell)

        softmax = Dense(units=num_classes,
                        kernel_initializer=he_uniform(),
                        activation='softmax')(rnn_dropout)

        model = Model(inputs=inputs, outputs=softmax)
        # opt = Adam(lr=CLASS_LRATE)
        opt = Adam(learning_rate=class_lrate)
        
        #Tarlis: removed the top_k metric in cases of less than 5 classes
        if self.config['num_classes'] < 5:
            _metrics = ['acc']
        else:
            _metrics = ['acc', 'top_k_categorical_accuracy']

        model.compile(optimizer=opt,
                           loss='categorical_crossentropy',
                           metrics=_metrics)
        
        return model

        
    def fit(self, 
            X_train, 
            y_train, 
            X_val,
            y_val, 
            save_results=False,
            res_path='.'):
        
        early_stopping_patience = self.config['early_stopping_patience']
        class_batch_size = self.config['class_batch_size']
        class_epochs = self.config['class_epochs']
        baseline_metric = self.config['baseline_metric']
        baseline_value = self.config['baseline_value']
        
        METRICS_FILE = None
        if save_results:
            METRICS_FILE = self.name+'-'+prefix+'_results.csv'
            METRICS_FILE = os.path.join(res_path, METRICS_FILE)
        
        self.model = self.create()
        
        history = self.model.fit(x=X_train,
                   y=y_train,
                   validation_data=(X_val, y_val),
                   batch_size=class_batch_size,
                   shuffle=True,
                   epochs=class_epochs,
                   verbose=self.config['verbose'],
                   callbacks=[EpochLogger(X_train, y_train, X_val, y_val, 
                                          metric=baseline_metric,
                                          baseline=baseline_value,
                                          patience=early_stopping_patience,
                                          verbose=1 if self.isverbose else 0,
                                          metrics_file=METRICS_FILE)])
        
        self.report = pd.DataFrame(history.history)
        return self.report
    
    def predict(self,                 
                X_test,
                y_test):
        
        y_pred = self.model.predict(X_test)
        
        # Because of one_hot_encoding:
        self.y_test_true = argmax(y_test, axis = 1)
        self.y_test_pred = argmax(y_pred , axis = 1)
        
        self._summary = self.score(self.y_test_true, y_pred)
        
        if self.le:
            self.y_test_true = self.le.inverse_transform(self.y_test_true)
            self.y_test_pred = self.le.inverse_transform(self.y_test_pred)

        return self._summary, y_pred

    
# --------------------------------------------------------------------------------
class EpochLogger(EarlyStopping):

    def __init__(self,X_train, y_train, 
                 X_test, y_test,
                 dataset='',
                 metric='val_acc', 
                 baseline=0,
                 patience=30,
                 metrics_file=None,
                 verbose=1):
        
        super(EpochLogger, self).__init__(monitor='val_acc',
                                          mode='max',
                                          patience=patience,
                                          verbose=verbose)
        
        self._metric = metric
        self._baseline = baseline
        self._baseline_met = False
        self.verbose = verbose
        
        self.dataset = dataset
        self.X_train = X_train
        self.y_train = y_train.argmax(axis=1)
        self.X_test = X_test
        self.y_test = y_test.argmax(axis=1)
        
        self._file = metrics_file
        
        self.metrics = MetricsLogger().load(metrics_file if metrics_file else 'metrics.csv')

    def on_epoch_begin(self, epoch, logs={}):
        if self.verbose:
            print("===== Training Epoch %d =====" % (epoch + 1))

        if self._baseline_met:
            super(EpochLogger, self).on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs={}):
        pred_y_train = np.array(self.model.predict(self.X_train))
        
        (train_acc,
         train_acc5,
         train_f1_macro,
         train_prec_macro,
         train_rec_macro, _) = compute_acc_acc5_f1_prec_rec(self.y_train,
                                                         pred_y_train,
                                                         print_metrics=True,
                                                         print_pfx='TRAIN')

        pred_y_test = np.array(self.model.predict(self.X_test))
        
        (test_acc,
         test_acc5,
         test_f1_macro,
         test_prec_macro,
         test_rec_macro, _) = compute_acc_acc5_f1_prec_rec(self.y_test,
                                                        pred_y_test,
                                                        print_metrics=True,
                                                        print_pfx='TEST')
        self.metrics.log('MARC', int(epoch + 1), self.dataset,
                    logs['loss'], train_acc, train_acc5,
                    train_f1_macro, train_prec_macro, train_rec_macro,
                    logs['val_loss'], test_acc, test_acc5,
                    test_f1_macro, test_prec_macro, test_rec_macro)
        if self._file:
            self.metrics.save(self._file)

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
            

###############################################################################
#   LOAD DATA
###############################################################################
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# from core.utils.geohash import bin_geohash

def loadTrajectories(train_file, test_file, logger=None):
    if logger:
        print('\n###########      DATA LOADING        ###########')
        print('TRAIN_FILE =', train_file)
        print('TEST_FILE =', test_file)
        logger.log(Logger.INFO, "Loading data from file(s) ... ")
    
#     df_train = pd.read_csv(train_file)
#     df_test = pd.read_csv(test_file)
    df_train = readDataset(os.path.dirname(train_file), file=os.path.basename(train_file), missing='-999')
    df_test = readDataset(os.path.dirname(test_file), file=os.path.basename(test_file), missing='-999')
    
    return df_train, df_test

def prepareTrajectories(df_train, df_test, tid_col='tid',
                     label_col='label', geo_precision=8, drop=[], logger=None):
    if logger:
        print('\n###########    DATA PREPARATION      ###########')
    
#     df_train = df_train.replace('?', np.nan)
#     df_test  = df_test.replace('?', np.nan)
    
    #df = df_train.copy().append(df_test)
    df = pd.concat([df_train, df_test])
    tids_train = df_train[tid_col].unique()

    keys = list(df.keys())
    vocab_size = {}
    keys.remove(tid_col)

    for col in drop:
        if col in keys:
            keys.remove(col)
            if logger:
                logger.log(Logger.WARNING, "Column '" + col + "' dropped " +
                       "from input file!")
        else:
            if logger:
                logger.log(Logger.WARNING, "Column '" + col + "' cannot be " +
                       "dropped because it was not found!")

    num_classes = len(set(df[label_col]))
    count_attr = 0
    lat_lon = False

    if 'lat' in keys and 'lon' in keys:
        keys.remove('lat')
        keys.remove('lon')
        lat_lon = True
        count_attr += geo_precision * 5
        if logger:
            logger.log(Logger.INFO, "Attribute Lat/Lon: " +
                   str(geo_precision * 5) + "-bits value")

    for attr in [item for item in keys if item != label_col]:
        
        df[attr] = LabelEncoder().fit_transform(df[attr].astype(str))
        vocab_size[attr] = max(df[attr]) + 1

        #if attr != label_col:
        values = len(set(df[attr]))
        count_attr += values
        if logger:
            logger.log(Logger.INFO, "Attribute '" + attr + "': " +
                       str(values) + " unique values")
    
    # for Label: 
    le = LabelEncoder()
    df[label_col] = le.fit_transform(df[label_col].astype(str))
    vocab_size[label_col] = max(df[label_col]) + 1

    if logger:
        logger.log(Logger.INFO, "Total of attribute/value pairs: " +
               str(count_attr))
    keys.remove(label_col)

    x = [[] for key in keys]
    y = []
    idx_train = []
    idx_test = []
    max_length = 0
    trajs = len(set(df[tid_col]))

    if lat_lon:
        x.append([])

    for idx, tid in enumerate(set(df[tid_col])):
        if logger:
            logger.log_dyn(Logger.INFO, "Processing trajectory " + str(idx + 1) +
                       "/" + str(trajs) + ". ")
        traj = df.loc[df[tid_col].isin([tid])]
        features = np.transpose(traj.loc[:, keys].values)

        for i in range(0, len(features)):
            x[i].append(features[i])

        if lat_lon:
            loc_list = []
            for i in range(0, len(traj)):
                lat = traj['lat'].values[i]
                lon = traj['lon'].values[i]
                loc_list.append(bin_geohash(lat, lon, geo_precision))
            x[-1].append(loc_list)

        label = traj[label_col].iloc[0]
        y.append(label)

        if tid in tids_train:
            idx_train.append(idx)
        else:
            idx_test.append(idx)

        if traj.shape[0] > max_length:
            max_length = traj.shape[0]

    if lat_lon:
        keys.append('lat_lon')
        vocab_size['lat_lon'] = geo_precision * 5

    one_hot_y = OneHotEncoder().fit(df.loc[:, [label_col]])

    x = [np.asarray(f) for f in x]
    y = one_hot_y.transform(pd.DataFrame(y)).toarray()
    if logger:
        logger.log(Logger.INFO, "Loading data from files ... DONE!")

#    x_train = np.asarray([f[idx_train] for f in x])
#    y_train = y[idx_train]
#    x_test = np.asarray([f[idx_test] for f in x])
#    y_test = y[idx_test]
    
    x_train = [f[idx_train] for f in x]
    y_train = y[idx_train]
    x_test = [f[idx_test] for f in x]
    y_test = y[idx_test]
    
    #x_train = np.asarray(x_train)
    #x_test = np.asarray(x_test)

    if logger:
        logger.log(Logger.INFO, 'Trajectories:  ' + str(trajs))
        logger.log(Logger.INFO, 'Labels:        ' + str(len(y[0])))
        logger.log(Logger.INFO, 'Train size:    ' + str(len(x_train[0]) / trajs))
        logger.log(Logger.INFO, 'Test size:     ' + str(len(x_test[0]) / trajs))
        
        #TODO: discover why this problem happen, maybe an error on preparing input (!important)
        #logger.log(Logger.INFO, 'x_train shape: ' + str(np.shape(x_train)))
        #logger.log(Logger.INFO, 'y_train shape: ' + str(y_train.shape))
        #logger.log(Logger.INFO, 'x_test shape:  ' + str(np.shape(x_test)))
        #logger.log(Logger.INFO, 'y_test shape:  ' + str(y_test.shape))

    return (keys, vocab_size, num_classes, max_length, le,
            x_train, x_test,
            y_train, y_test)