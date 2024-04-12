# -*- coding: utf-8 -*-
'''
MAT-analysis: Analisys and Classification methods for Multiple Aspect Trajectory Data Mining

The present package offers a tool, to support the user in the task of data analysis of multiple aspect trajectories. It integrates into a unique framework for multiple aspects trajectories and in general for multidimensional sequence data mining methods.
Copyright (C) 2022, MIT license (this portion of code is subject to licensing from source project distribution)

Created on Dec, 2021
Copyright (C) 2022, License GPL Version 3 or superior (this portion of code is subject to licensing from source project distribution)

@author: Tarlis Portela (adapted)

# Original source:
# Author: Nicksson C. A. de Freitas, 
          Ticiana L. Coelho da Silva, 
          Jose António Fernandes de Macêdo, 
          Leopoldo Melo Junior, 
          Matheus Gomes Cordeiro
# Adapted from: https://github.com/nickssonfreitas/ICAART2021
'''
# --------------------------------------------------------------------------------
import os
from os import path
import pandas as pd
import numpy as np

from tqdm.auto import tqdm
from datetime import datetime
from glob2 import glob
import json
import mplleaflet as mpl
import traceback
import time
import gc
import itertools
import collections
from joblib import load, dump

# --------------------------------------------------------------------------------
from matanalysis.methods._lib.datahandler import loadTrajectories, prepareTrajectories
from matanalysis.methods._lib.utils import update_report, print_params, concat_params
# --------------------------------------------------------------------------------
from matanalysis.methods._lib.pymove.core import utils
from matanalysis.methods._lib.pymove.models.classification import Tulvae as tva
from matanalysis.methods._lib.models import ModelContainer
# --------------------------------------------------------------------------------

def TULVAE_read(dir_path, res_path='.', prefix='', save_results=False, n_jobs=-1, random_state=42, rounds=10, label_poi='poi', geohash=False, geo_precision=30):
    
    # Load Data - Tarlis:
    df_train, df_test = loadTrajectories(dir_path, prefix)
    
    return TULVAE(df_train, df_test, res_path, prefix, save_results, n_jobs, random_state, rounds, label_poi, geohash, geo_precision)
    

def TULVAE(df_train, df_test, res_path='.', prefix='', save_results=False, n_jobs=-1, random_state=42, rounds=10, label_poi='poi', geohash=False, geo_precision=30):
    
    dir_validation = os.path.join(res_path, 'TULVAE-'+prefix, 'validation')
    dir_evaluation = os.path.join(res_path, 'TULVAE-'+prefix)
    
    # Load Data - Tarlis:
    X, y, features, num_classes, space, dic_parameters = prepareTrajectories(df_train.copy(), df_test.copy(),
                                                                          split_test_validation=True,
                                                                          features_encoding=True, 
                                                                          y_one_hot_encodding=False,
                                                                          data_preparation=2,
                                                                          features=[label_poi],
                                                                          space_geohash=geohash,
                                                                          geo_precision=geo_precision)
    
    assert (len(X) > 2), "[TULVAE:] ERR: data is not set or < 3"
    if len(X) > 2:
        X_train = X[0]
        X_val   = X[1]
        X_test  = X[2]
        y_train = y[0]
        y_val   = y[1]
        y_test  = y[2]

    #df_train = pd.read_csv(file_train)
    #df_val = pd.read_csv(file_val)
    #df_test = pd.read_csv(file_test)
    #df = pd.concat([df_train, df_val, df_test])

    #le = LabelEncoder()
    #df['day'] = le.fit_transform(df['day'])
    #df_train['day'] = le.transform(df_train['day'])
    #df_val['day'] = le.transform(df_val['day'])
    #df_test['day'] = le.transform(df_test['day'])
    
    #if 'day' in X_train.columns:
    #    le = LabelEncoder()
    #    X_train['day'] = le.fit_transform(X_train['day'])
    #    X_val['day'] = le.fit_transform(X_val['day'])
    #    X_test['day'] = le.fit_transform(X_test['day'])


    ## ## Get trajectories
#
    #label_poi = 'poi'
    #features = ['poi', 'label', 'tid']
    #data = [df_train[features], df_val[features], df_test[features]]
#
#
    #X, y, dic_parameters = datautils.generate_X_y_rnn(data=data,
    #                           features_encoding=True,
    #                           y_one_hot_encodding=False,
    #                           label_y='label',
    #                           label_segment='tid')


    # ## GRID SEARCH TO TULER
    #num_classes = dic_parameters['num_classes']
    max_lenght = dic_parameters['max_lenght']
    vocab_size = dic_parameters['vocab_size'][label_poi] #['poi']
    rnn= ['bilstm']
    units = [100, 200, 300]
    stack = [1]
    dropout =[0.5]
    embedding_size = [100, 200, 300]
    z_values = [100,200,300]
    batch_size = [64]
    epochs = [1000]
    patience = [20]
    monitor = ['val_acc']
    optimizer = ['ada']
    learning_rate = [0.001]
            
    print("\n[TULVAE:] Building TULVAE Model")
    start_time = datetime.now()

    total = len(rnn)*len(units)*len(stack)* len(dropout)* len(embedding_size)* len(z_values) * len(batch_size)*len(epochs) * len(patience) *len(monitor) * len(learning_rate) 
    print('[TULVAE:] Starting model training, {} iterations'.format(total))
    
    if save_results and not os.path.exists(dir_validation):
        os.makedirs(dir_validation)
    
    # Hiper-param data:
    data = []
    def getParamData(f):
        marksplit = '-'
        df_ = pd.read_csv(f)
        f = f[f.find('tulvae-'):]
        df_['nn']=   f.split(marksplit)[1]
        df_['un']=     f.split(marksplit)[2]
        df_['st']=     f.split(marksplit)[3]
        df_['dp'] = f.split(marksplit)[4]
        df_['es'] = f.split(marksplit)[5]
        df_['zv'] = f.split(marksplit)[6]
        df_['bs'] = f.split(marksplit)[7]
        df_['epoch'] = f.split(marksplit)[8]
        df_['pat'] = f.split(marksplit)[9]
        df_['mon'] = f.split(marksplit)[10]
        df_['lr'] = f.split(marksplit)[11]
        df_['fet'] = f.split(marksplit)[12].split('.csv')[0]
        data.append(df_)

    pbar = tqdm(itertools.product(rnn, units, stack, dropout, embedding_size, 
                                  z_values, batch_size,epochs, patience, monitor, learning_rate), 
                total=total, desc="[TULVAE:] Model Training")
    for c in pbar:
        nn=c[0]
        un=c[1]
        st=c[2]
        dp=c[3]
        es=c[4]
        zv=c[5]
        bs=c[6]
        epoch=c[7]
        pat=c[8]
        mon=c[9]
        lr=c[10]

        filename = os.path.join(dir_validation, 'tulvae-'+
                                concat_params(nn, un, st, dp, es, zv, bs, epoch, pat, mon, lr, features)+'.csv')

        if os.path.exists(filename):
            pbar.set_postfix_str('Skip ---> {}\n'.format(filename))
            getParamData(filename)
        else:

            pbar.set_postfix_str(print_params('nn, un, st, dp, es, zv, bs, epoch, pat, mon, lr',
                                             nn, un, st, dp, es, zv, bs, epoch, pat, mon, lr))
            #try:
            tulvae = tva.TulvaeClassier(max_lenght=max_lenght,    
                        num_classes=num_classes,
                        vocab_size=vocab_size,
                        rnn_units=un,
                        dropout=dp,
                        embedding_size=es,
                        z_values=zv,
                        stack=st)

            tulvae.fit(X_train, y_train,
                        X_val, y_val,
                        batch_size=bs,
                        epochs=epoch,
                        learning_rate=lr,
                        save_model=False,
                        save_best_only=False,
                        save_weights_only=False)

            validation_report, y_pred = tulvae.predict(X_val, y_val)

            if save_results:
                validation_report.to_csv(filename, index=False)


            data.append( update_report(validation_report, 'nn, un, st, dp, es, zv, bs, epoch, pat, mon, lr, features',
                                       nn, un, st, dp, es, zv, bs, epoch, pat, mon, lr, features) )

            tulvae.free()
            #except:
            #    print('[TULVAE:] Error training - '+
            #          print_params('nn, un, st, dp, es, zv, bs, epoch, pat, mon, lr',
            #                       nn, un, st, dp, es, zv, bs, epoch, pat, mon, lr))
            #    #pass


    df_result = pd.concat(data)
    df_result.reset_index(drop=True, inplace=True)
    df_result.sort_values('acc', ascending=False, inplace=True)

    model = 0
    nn = df_result.iloc[model]['nn']
    un = int(df_result.iloc[model]['un'])
    st = int(df_result.iloc[model]['st'])
    dp = float(df_result.iloc[model]['dp'])
    es = int(df_result.iloc[model]['es'])
    zv = int(df_result.iloc[model]['es'])
    bs = int(df_result.iloc[model]['bs'])
    epoch = int(df_result.iloc[model]['epoch'])
    pat = int(df_result.iloc[model]['pat'])
    mon = df_result.iloc[model]['mon']
    lr = float(df_result.iloc[0]['lr'])

    filename = os.path.join(dir_evaluation, 'eval_tuvae-'+concat_params(nn, un, st, dp, es, zv, bs, epoch, pat, mon, lr, features)+'.csv')

    print("[TULVAE:] Filename: {}.".format(filename))

    if not os.path.exists(filename):
        print('[TULVAE:] Creating a model to test set')
        print("[TULVAE:] Parameters: " + print_params('nn, un, st, dp, es, zv, bs, epoch, pat, mon, lr, features',
                                              nn, un, st, dp, es, zv, bs, epoch, pat, mon, lr, features) )

        evaluate_report = []
#        rounds = 10

        pbar = tqdm(range(rounds), desc="Model Testing")
        for e in pbar:
            pbar.set_postfix_str('Rounds {} of {}'.format(e, rounds))
            tulvae = tva.TulvaeClassier(max_lenght=max_lenght,    
                        num_classes=num_classes,
                        vocab_size=vocab_size,
                        rnn_units=un,
                        dropout=dp,
                        embedding_size=es,
                        z_values=zv,
                        stack=st)

            tulvae.fit(X_train, y_train,
                        X_val, y_val,
                        batch_size=bs,
                        epochs=epoch,
                        learning_rate=lr,
                        save_model=False,
                        save_best_only=False,
                        save_weights_only=False)

            #evaluate_report.append(tulvae.predict(X_test, y_test))
            eval_report, y_pred = tulvae.predict(X_test, y_test)
            evaluate_report.append(eval_report)
            ###tulvae.free()
            
        evaluate_report = pd.concat(evaluate_report)
        if save_results:
            if not os.path.exists(dir_evaluation):
                os.makedirs(dir_evaluation)
                
            evaluate_report.to_csv(filename, index=False)
            
        end_time = (datetime.now()-start_time).total_seconds() * 1000
        print('[TULVAE:] Processing time: {} milliseconds. Done.'.format(end_time))
        #return evaluate_report
        # ---------------------------------------------------------------------------------
        # Prediction
        # ---------------------------------------------------------------------------------
        model = ModelContainer(tulvae, y_test, X_test, cls_history=evaluate_report, approach='TULVAE', le=dic_parameters['encode_y'])

#        if save_results:
#            model.save(dir_path, modelfolder)

        return model
    else:
        print('[TULVAE:] Model previoulsy built.')
        
    print('\n--------------------------------------\n')