# -*- coding: utf-8 -*-
'''
MAT-analysis: Analisys and Classification methods for Multiple Aspect Trajectory Data Mining

The present package offers a tool, to support the user in the task of data analysis of multiple aspect trajectories. It integrates into a unique framework for multiple aspects trajectories and in general for multidimensional sequence data mining methods.
Copyright (C) 2022, MIT license (this portion of code is subject to licensing from source project distribution)

Created on Dec, 2021
Copyright (C) 2022, License GPL Version 3 or superior (see LICENSE file)

@author: Tarlis Portela
@author: Lucas May Petry
'''
# --------------------------------------------------------------------------------
import os 
import numpy as np
import pandas as pd
from datetime import datetime
#sys.path.insert(0, os.path.abspath('.')) # TODO fix imports

# --------------------------------------------------------------------------------
from numpy import argmax
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
  
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout
from tensorflow.keras.initializers import he_uniform
from tensorflow.keras.regularizers import l1
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Add, Average, Concatenate, Embedding
from tensorflow.keras.callbacks import EarlyStopping
#from core.utils.metrics import compute_acc_acc5_f1_prec_rec 
from tensorflow import random
from sklearn.metrics import classification_report
#from automatize.methods._lib.metrics import classification_report_csv
# --------------------------------------------------------------------------------
from matdata.preprocess import readDataset
#from core.utils.geohash import bin_geohash
from matanalysis.methods._lib.geohash import bin_geohash
#from core.logger import Logger
from matanalysis.methods._lib.logger import Logger
#from core.utils.metrics import MetricsLogger
from matanalysis.methods._lib.metrics import MetricsLogger
from matanalysis.methods._lib.metrics import classification_report_dict2csv, classification_report_dict2df
from matanalysis.methods._lib.metrics import compute_acc_acc5_f1_prec_rec
from matanalysis.methods._lib.models import ModelContainer
# --------------------------------------------------------------------------------

def MARC_read(train_file, test_file, res_path='.', prefix='', dataset='', save_results=False, n_jobs=-1, random_state=42, geo_precision=8, embedder_size=100, merge_type='concatenate', rnn_cell='lstm'):
    
    df_train, df_test = loadTrajectories(train_file, test_file)
    
    return MARC(df_train, df_test, res_path, prefix, dataset, save_results, random_seed, geo_precision, embedder_size, merge_type, rnn_cell)

def MARC(df_train, df_test, res_path='.', prefix='', dataset='', save_results=False, n_jobs=-1, random_state=42, geo_precision=8, embedder_size=100, merge_type='concatenate', rnn_cell='lstm'):
    
    return marc_model(df_train, df_test, res_path, prefix, dataset, save_results, n_jobs, random_state, geo_precision, embedder_size, merge_type, rnn_cell)

def marc_model(df_train, df_test, res_path='.', prefix='', dataset='', save_results=False, n_jobs=-1, random_state=42, geo_precision=8, embedder_size=100, merge_type='concatenate', rnn_cell='lstm'):

    np.random.seed(seed=random_state)
    random.set_seed(random_state)
    #random.seed(random_seed)
    
    logger = Logger()

    METHOD = 'MARC'
    METRICS_FILE = 'MARC-'+prefix+'_results.csv'
    METRICS_FILE = os.path.join(res_path, METRICS_FILE)
                                            
    CLASS_DROPOUT = 0.5
    CLASS_HIDDEN_UNITS = 100
    CLASS_LRATE = 0.001
    CLASS_BATCH_SIZE = 64
    CLASS_EPOCHS = 1000
    EARLY_STOPPING_PATIENCE = 30
    BASELINE_METRIC = 'acc'
    BASELINE_VALUE = 0.5
    
    print("[MARC:] Starting: building neural network for", METHOD, dataset)
    # from datetime import datetime
    time = datetime.now()

    VALID_MERGES = ['add', 'average', 'concatenate']
    VALID_CELLS = ['lstm', 'gru']

    if merge_type not in VALID_MERGES:
        print("Merge type '" + merge_type + "' is not valid!\n",
              "Please choose 'add', 'average', or 'concatenate'.")
        exit()

    if rnn_cell not in VALID_CELLS:
        print("RNN cell type '" + rnn_cell + "' is not valid!\n",
              "Please choose 'lstm' or 'gru'.")
        exit()


    print('====================================', 'PARAMS',
          '====================================')
#    print('TRAIN_FILE =', TRAIN_FILE)
#    print('TEST_FILE =', TEST_FILE)
    print('METRICS_FILE =', METRICS_FILE)
    print('dataset =', dataset)
    print('embedder_size =', embedder_size)
    print('merge_type =', merge_type, '\n')

    # from core.utils.metrics import MetricsLogger
    metrics = MetricsLogger().load(METRICS_FILE)


    (keys, vocab_size,
     num_classes,
     max_length,
     le,
     x_train, x_test,
     y_train, y_test) = prepareTrajectories(df_train.copy(), df_test.copy(),
                                         tid_col='tid',
                                         label_col='label',
                                         logger=logger,
                                         geo_precision=geo_precision)

    ###############################################################################
    #   PREPARING CLASSIFIER DATA
    ###############################################################################
    # from keras.preprocessing.sequence import pad_sequences


    cls_x_train = [pad_sequences(f, max_length, padding='pre') for f in x_train]
    cls_x_test = [pad_sequences(f, max_length, padding='pre') for f in x_test]
    cls_y_train = y_train
    cls_y_test = y_test


    ###############################################################################
    #   CLASSIFIER
    ###############################################################################
    # from keras.models import Model
    # from keras.layers import Dense, LSTM, GRU, Dropout
    # from keras.initializers import he_uniform
    # from keras.regularizers import l1
    # from keras.optimizers import Adam
    # from keras.layers import Input, Add, Average, Concatenate, Embedding
    # from keras.callbacks import EarlyStopping
    # from core.utils.metrics import compute_acc_acc5_f1_prec_rec


#    CLASS_DROPOUT = 0.5
#    CLASS_HIDDEN_UNITS = 100
#    CLASS_LRATE = 0.001
#    CLASS_BATCH_SIZE = 64
#    CLASS_EPOCHS = 1000
#    EARLY_STOPPING_PATIENCE = 30
#    BASELINE_METRIC = 'acc'
#    BASELINE_VALUE = 0.5


    print('=====================================', METHOD,
          '=====================================')

    print('CLASS_DROPOUT =', CLASS_DROPOUT)
    print('CLASS_HIDDEN_UNITS =', CLASS_HIDDEN_UNITS)
    print('CLASS_LRATE =', CLASS_LRATE)
    print('CLASS_BATCH_SIZE =', CLASS_BATCH_SIZE)
    print('CLASS_EPOCHS =', CLASS_EPOCHS)
    print('EARLY_STOPPING_PATIENCE =', EARLY_STOPPING_PATIENCE)
    print('BASELINE_METRIC =', BASELINE_METRIC)
    print('BASELINE_VALUE =', BASELINE_VALUE, '\n')


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
            pred_y_train = np.array(self.model.predict(cls_x_train))
            (train_acc,
             train_acc5,
             train_f1_macro,
             train_prec_macro,
             train_rec_macro) = compute_acc_acc5_f1_prec_rec(cls_y_train,
                                                             pred_y_train,
                                                             print_metrics=True,
                                                             print_pfx='TRAIN')

            pred_y_test = np.array(self.model.predict(cls_x_test))
            (test_acc,
             test_acc5,
             test_f1_macro,
             test_prec_macro,
             test_rec_macro) = compute_acc_acc5_f1_prec_rec(cls_y_test,
                                                            pred_y_test,
                                                            print_metrics=True,
                                                            print_pfx='TEST')
            metrics.log(METHOD, int(epoch + 1), dataset,
                        logs['loss'], train_acc, train_acc5,
                        train_f1_macro, train_prec_macro, train_rec_macro,
                        logs['val_loss'], test_acc, test_acc5,
                        test_f1_macro, test_prec_macro, test_rec_macro)
            if save_results:
                metrics.save(METRICS_FILE)

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

    hidden_dropout = Dropout(CLASS_DROPOUT)(hidden_input)

    if rnn_cell == 'lstm':
        rnn_cell = LSTM(units=CLASS_HIDDEN_UNITS,
                        recurrent_regularizer=l1(0.02))(hidden_dropout)
    else:
        rnn_cell = GRU(units=CLASS_HIDDEN_UNITS,
                       recurrent_regularizer=l1(0.02))(hidden_dropout)

    rnn_dropout = Dropout(CLASS_DROPOUT)(rnn_cell)

    softmax = Dense(units=num_classes,
                    kernel_initializer=he_uniform(),
                    activation='softmax')(rnn_dropout)

    classifier = Model(inputs=inputs, outputs=softmax)
    # opt = Adam(lr=CLASS_LRATE)
    opt = Adam(learning_rate=CLASS_LRATE)

    classifier.compile(optimizer=opt,
                       loss='categorical_crossentropy',
                       metrics=['acc', 'top_k_categorical_accuracy'])

    history = classifier.fit(x=cls_x_train,
                   y=cls_y_train,
                   validation_data=(cls_x_test, cls_y_test),
                   batch_size=CLASS_BATCH_SIZE,
                   shuffle=True,
                   epochs=CLASS_EPOCHS,
                   verbose=0,
                   callbacks=[EpochLogger(metric=BASELINE_METRIC,
                                          baseline=BASELINE_VALUE)])

    # Prediction
    # ---------------------------------------------------------------------------------
    model = ModelContainer(classifier, cls_y_test, cls_x_test, cls_history=history.history, approach='MARC', le=le)
    
    if save_results:
        model.save(dir_path, modelfolder)
    
#    y_test_true_dec = le.inverse_transform(argmax( cls_y_test, axis = 1)) # le.inverse_transform(argmax(y_test1, axis = 1))
#    y_test_pred_dec = le.inverse_transform(argmax( classifier.predict(cls_x_test), axis = 1)) # le.inverse_transform(argmax( classifier.predict(X_test) , axis = 1))
#
#    report = classification_report(y_test_true_dec, y_test_pred_dec, output_dict=True, zero_division=False)
##    report = classification_report_dict2df(report, "MARC")  
#    # ---------------------------------------------------------------------------------
#    if (save_results) :
#        dir_path = os.path.dirname(METRICS_FILE)
#        #if not os.path.exists(os.path.join(dir_path, modelfolder)):
#        #    os.makedirs(os.path.join(dir_path, modelfolder))
#        #classifier.save(os.path.join(dir_path, modelfolder, 'model_approach1.h5'))
#
#        classification_report_dict2csv(report, os.path.join(dir_path, 'model_marc_report.csv'),"MARC")  
##        dataframe.to_csv(os.path.join(dir_path, 'model_marc_report.csv'), index = False)
#        pd.DataFrame(history.history).to_csv(os.path.join(dir_path, "model_marc_history.csv"))
#        pd.DataFrame(y_test_true_dec,y_test_pred_dec).to_csv(os.path.join(dir_path, 'model_marc_prediction.csv'), header=['true_label'], index_label='prediction')
    
    time_ext = (datetime.now()-time).total_seconds() * 1000
    print(f"Processing time: {time_ext} milliseconds. Done.")
    print('------------------------------------------------------------------------------------------------')
    
    return model


###############################################################################
#   LOAD DATA
###############################################################################
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# from core.utils.geohash import bin_geohash

def loadTrajectories(train_file, test_file):
    print('\n###########      DATA LOADING        ###########')
    print('TRAIN_FILE =', train_file)
    print('TEST_FILE =', test_file)
    if logger:
        logger.log(Logger.INFO, "Loading data from file(s) ... ")
    
#     df_train = pd.read_csv(train_file)
#     df_test = pd.read_csv(test_file)
    df_train = readDataset(os.path.dirname(train_file), file=os.path.basename(train_file), missing='-999')
    df_test = readDataset(os.path.dirname(test_file), file=os.path.basename(test_file), missing='-999')
    
    return df_train, df_test

def prepareTrajectories(df_train, df_test, tid_col='tid',
                     label_col='label', geo_precision=8, drop=[], logger=None):
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
