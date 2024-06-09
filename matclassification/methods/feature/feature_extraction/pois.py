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

from datetime import datetime
# --------------------------------------------------------------------------------
from matdata.preprocess import readDataset, dfVariance
from matclassification.methods._lib.geohash import bin_geohash

## POI-S: POI Sequence (POI-F extension) [By Tarlis]
## --------------------------------------------------------------------------------------------

def pois_read(sequences, features, method='npoi', dataset='specific', folder='./data', result_dir='.', save_all=False, tid_col='tid', class_col='label'):
    
    df_train, df_test = loadTrainTest(features, folder, dataset)
    
    return pois(df_train, df_test, sequences, features, method, result_dir, save_all, tid_col, class_col)

def pois(df_train, df_test, sequences, features, method='npoi', result_dir='.', save_all=False, tid_col='tid', class_col='label', verbose=True):
    
    if verbose:
        print("[POIS:] Starting feature extractor ... ")
    time = datetime.now()
    
    if features is None:
        df_ = df_train.copy()
        stats = dfVariance(df_[[x for x in df_.columns if x not in [tid_col, class_col]]])
        features = [stats.index[0]]
    
#    if save_all:
#        save_all = result_dir
        
    agg_x_train = None
    agg_x_test  = None
    
    for sequence in sequences:
        aux_x_train = None
        aux_x_test  = None
        for feature in features:
            if verbose:
                print('- Feature: {}, Sequence: {}'.format(feature, sequence))
            unique_features = df_train[feature].unique().tolist()

            points = df_train[feature].values
            possible_sequences = []
            for idx in range(0, (len(points)-(sequence - 1))):
                aux = []
                for i in range (0, sequence):
                    aux.append(points[idx + i])
                aux = tuple(aux)
                if aux not in possible_sequences:
                    possible_sequences.append(aux)

            seq2idx = dict(zip(possible_sequences, np.r_[0:len(possible_sequences)]))

            if save_all:
                if not os.path.exists(result_dir):
                    os.makedirs(result_dir)

                pd.DataFrame(possible_sequences).to_csv(os.path.join(result_dir, \
                   feature+'_'+str(sequence)+'-sequences.csv'), index=False, header=None)
            
            if method == 'poi':
                if verbose:
                    print('Starting POI...')
                x_train, x_test, y_train, y_test = poi(df_train, df_test, possible_sequences, \
                                                       seq2idx, sequence, feature, result_dir=None, 
                                                       tid_col=tid_col, class_col=class_col)
            elif method == 'npoi':
                if verbose:
                    print('Starting NPOI...')
                x_train, x_test, y_train, y_test = npoi(df_train, df_test, possible_sequences, \
                                                       seq2idx, sequence, feature, result_dir=None, 
                                                       tid_col=tid_col, class_col=class_col)
            else:
                if verbose:
                    print('Starting WNPOI...')
                x_train, x_test, y_train, y_test = wnpoi(df_train, df_test, possible_sequences, \
                                                       seq2idx, sequence, feature, result_dir=None, 
                                                       tid_col=tid_col, class_col=class_col)

            # Concat columns:
            if aux_x_train is None:
                aux_x_train = pd.DataFrame(x_train)
            else:
                aux_x_train = pd.concat([aux_x_train, pd.DataFrame(x_train)], axis=1)   

            if aux_x_test is None:
                aux_x_test = pd.DataFrame(x_test)
            else:
                aux_x_test = pd.concat([aux_x_test, pd.DataFrame(x_test)], axis=1)    
                
        # Write features concat:
        if save_all:
            core_name = os.path.join(result_dir, method+'_'+('_'.join(features))+'_'+('_'.join([str(sequence)])) ) #+'_'+dataset)
            to_file(core_name, aux_x_train, aux_x_test, y_train, y_test)
        
        if agg_x_train is None:
            agg_x_train = aux_x_train
        else:
            agg_x_train = pd.concat([agg_x_train, aux_x_train], axis=1)   

        if agg_x_test is None:
            agg_x_test = aux_x_test
        else:
            agg_x_test = pd.concat([agg_x_test, aux_x_test], axis=1)    
                
    
    del df_train
    del df_test 
    del x_train
    del x_test   
   
    core_name = os.path.join(result_dir, method+'_'+('_'.join(features))+'_'+('_'.join([str(n) for n in sequences])) ) #+'_'+dataset)
    if save_all:
        to_file(core_name, agg_x_train, agg_x_test, y_train, y_test)

    time_ext = (datetime.now()-time).total_seconds() * 1000
    if verbose:
        print('[POIS:] Processing time: {} milliseconds. Done.'.format(time_ext))
        print('------------------------------------------------------------------------------------------------')
    
    return agg_x_train, agg_x_test, y_train, y_test, core_name
    
## --------------------------------------------------------------------------------------------
## POI-F: POI Frequency
def poi(df_train, df_test, possible_sequences, seq2idx, sequence, feature, result_dir=None, tid_col='tid', class_col='label'):
#     from ..main import importer
#     importer(['S'], locals())

#    print('Starting POI...')
    method = 'poi'
    
    # Train
    train_tids = df_train[tid_col].unique()
    x_train = np.zeros((len(train_tids), len(possible_sequences)))
    y_train = df_train.drop_duplicates(subset=[tid_col, class_col],
                                       inplace=False) \
                      .sort_values(tid_col, ascending=True,
                                   inplace=False)[class_col].values

    for i, tid in enumerate(train_tids):
        traj_pois = df_train[df_train[tid_col] == tid][feature].values
        for idx in range(0, (len(traj_pois)-(sequence - 1))):
            aux = []
            for b in range (0, sequence):
                aux.append(traj_pois[idx + b])
            aux = tuple(aux)
            x_train[i][seq2idx[aux]] += 1

    # Test
    test_tids = df_test[tid_col].unique()
    test_unique_features = df_test[feature].unique().tolist()
    x_test = np.zeros((len(test_tids), len(possible_sequences)))
    y_test = df_test.drop_duplicates(subset=[tid_col, class_col],
                                       inplace=False) \
                      .sort_values(tid_col, ascending=True,
                                   inplace=False)[class_col].values

    for i, tid in enumerate(test_tids):
        traj_pois = df_test[df_test[tid_col] == tid][feature].values
        for idx in range(0, (len(traj_pois)-(sequence - 1))):
            aux = []
            for b in range (0, sequence):
                aux.append(traj_pois[idx + b])
            aux = tuple(aux)
            if aux in possible_sequences:
                x_test[i][seq2idx[aux]] += 1
    
    if result_dir:
        core_name = os.path.join(result_dir, method+'_'+feature+'_'+str(sequence))
        to_file(core_name, x_train, x_test, y_train, y_test)
        
    return x_train, x_test, y_train, y_test
    
### NPOI-F: Normalized POI Frequency
def npoi(df_train, df_test, possible_sequences, seq2idx, sequence, feature, result_dir=None, tid_col='tid', class_col='label'):
#     from ..main import importer
#     importer(['S'], locals())
    
#    print('Starting NPOI...')
    method = 'npoi'
    
    # Train
    train_tids = df_train[tid_col].unique()
    x_train = np.zeros((len(train_tids), len(possible_sequences)))
    y_train = df_train.drop_duplicates(subset=[tid_col, class_col],
                                       inplace=False) \
                      .sort_values(tid_col, ascending=True,
                                   inplace=False)[class_col].values

    for i, tid in enumerate(train_tids):
        traj_pois = df_train[df_train[tid_col] == tid][feature].values
        for idx in range(0, (len(traj_pois)-(sequence - 1))):
            aux = []
            for b in range (0, sequence):
                aux.append(traj_pois[idx + b])
            aux = tuple(aux)
            x_train[i][seq2idx[aux]] += 1
        x_train[i] = x_train[i]/len(traj_pois)

    # Test
    test_tids = df_test[tid_col].unique()
    test_unique_features = df_test[feature].unique().tolist()
    x_test = np.zeros((len(test_tids), len(possible_sequences)))
    y_test = df_test.drop_duplicates(subset=[tid_col, class_col],
                                       inplace=False) \
                      .sort_values(tid_col, ascending=True,
                                   inplace=False)[class_col].values

    for i, tid in enumerate(test_tids):
        traj_pois = df_test[df_test[tid_col] == tid][feature].values
        for idx in range(0, (len(traj_pois)-(sequence - 1))):
            aux = []
            for b in range (0, sequence):
                aux.append(traj_pois[idx + b])
            aux = tuple(aux)
            if aux in possible_sequences:
                x_test[i][seq2idx[aux]] += 1
        x_test[i] = x_test[i]/len(traj_pois)
        
    if result_dir:
        core_name = os.path.join(result_dir, method+'_'+feature+'_'+str(sequence))
        to_file(core_name, x_train, x_test, y_train, y_test)
        
    return x_train, x_test, y_train, y_test
    
### WNPOI-F: Weighted Normalized POI Frequency.
def wnpoi(df_train, df_test, possible_sequences, seq2idx, sequence, feature, result_dir=None, tid_col='tid', class_col='label'):
#     from ..main import importer
#     importer(['S'], locals())
    
#    print('Starting WNPOI...')    
    method = 'wnpoi'
    
    train_labels = df_train[class_col].unique()
    weights = np.zeros(len(possible_sequences))
    for label in train_labels:
        aux_w = np.zeros(len(possible_sequences))
        class_pois = df_train[df_train[class_col] == label][feature].values
        for idx in range(0, (len(class_pois)-(sequence - 1))):
            aux = []
            for b in range (0, sequence):
                aux.append(class_pois[idx + b])
            aux = tuple(aux)
            seqidx = seq2idx[aux]
            if aux_w[seqidx] == 0:
                weights[seqidx] += 1
                aux_w[seqidx] = 1
    weights = np.log2(len(train_labels)/weights)
    # Train
    train_tids = df_train[tid_col].unique()
    x_train = np.zeros((len(train_tids), len(possible_sequences)))
    y_train = df_train.drop_duplicates(subset=[tid_col, class_col],
                                       inplace=False) \
                      .sort_values(tid_col, ascending=True,
                                   inplace=False)[class_col].values

    for i, tid in enumerate(train_tids):
        traj_pois = df_train[df_train[tid_col] == tid][feature].values
        for idx in range(0, (len(traj_pois)-(sequence - 1))):
            aux = []
            for b in range (0, sequence):
                aux.append(traj_pois[idx + b])
            aux = tuple(aux)
            x_train[i][seq2idx[aux]] += 1
        x_train[i] = x_train[i]/len(traj_pois)
        for w in range(0, len(possible_sequences)):
            x_train[i][w] *= weights[w]

    # Test
    test_tids = df_test[tid_col].unique()
    test_unique_features = df_test[feature].unique().tolist()
    x_test = np.zeros((len(test_tids), len(possible_sequences)))
    y_test = df_test.drop_duplicates(subset=[tid_col, class_col],
                                       inplace=False) \
                      .sort_values(tid_col, ascending=True,
                                   inplace=False)[class_col].values

    for i, tid in enumerate(test_tids):
        traj_pois = df_test[df_test[tid_col] == tid][feature].values
        for idx in range(0, (len(traj_pois)-(sequence - 1))):
            aux = []
            for b in range (0, sequence):
                aux.append(traj_pois[idx + b])
            aux = tuple(aux)
            if aux in possible_sequences:
                x_test[i][seq2idx[aux]] += 1
        x_test[i] = x_test[i]/len(traj_pois)
        for w in range(0, len(possible_sequences)):
            x_test[i][w] *= weights[w]
            
    if result_dir:
        core_name = os.path.join(result_dir, method+'_'+feature+'_'+str(sequence))
        to_file(core_name, x_train, x_test, y_train, y_test)
        
    return x_train, x_test, y_train, y_test
    
## --------------------------------------------------------------------------------------------
def poifreq_all(sequence, dataset, feature, folder, result_dir, tid_col='tid', class_col='label'):
#     from ..main import importer
#     importer(['S'], locals())
    print('Dataset: {}, Feature: {}, Sequence: {}'.format(dataset, feature, sequence))
#     df_train = pd.read_csv(folder+dataset+'_train.csv')
#     df_test = pd.read_csv(folder+dataset+'_test.csv')
    
    df_train, df_test = loadTrainTest([feature], folder, dataset)
    
    unique_features = df_train[feature].unique().tolist()
    
    points = df_train[feature].values
    possible_sequences = []
    for idx in range(0, (len(points)-(sequence - 1))):
        aux = []
        for i in range (0, sequence):
            aux.append(points[idx + i])
        aux = tuple(aux)
        if aux not in possible_sequences:
            possible_sequences.append(aux)

    seq2idx = dict(zip(possible_sequences, np.r_[0:len(possible_sequences)]))
    
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        
    pd.DataFrame(possible_sequences).to_csv(os.path.join(result_dir, feature+'_'+str(sequence)+'-sequences.csv'), index=False, header=None)
    
    poi(df_train, df_test, possible_sequences, seq2idx, sequence, feature, result_dir, tid_col, class_col)
    npoi(df_train, df_test, possible_sequences, seq2idx, sequence, feature, result_dir, tid_col, class_col)
    wnpoi(df_train, df_test, possible_sequences, seq2idx, sequence, feature, result_dir, tid_col, class_col)
   
    
## --------------------------------------------------------------------------------------------
def to_file(core_name, x_train, x_test, y_train, y_test):
#     from ..main import importer
#     importer(['pd'], locals())
    df_x_train = pd.DataFrame(x_train).to_csv(core_name+'-x_train.csv', index=False)#, header=None)
    df_x_test = pd.DataFrame(x_test).to_csv(core_name+'-x_test.csv', index=False)#, header=None)
    df_y_train = pd.DataFrame(y_train, columns=['label']).to_csv(core_name+'-y_train.csv', index=False)
    df_y_test = pd.DataFrame(y_test, columns=['label']).to_csv(core_name+'-y_test.csv', index=False)
    
def geoHasTransform(df, geo_precision=8):
#     from ..main import importer
#    importer(['geohash'], globals()) #globals
#     from ensemble_models.utils import geohash
    return [geohash(df['lat'].values[i], df['lon'].values[i], geo_precision) for i in range(0, len(df))]

def loadTrainTest(features, folder, dataset=''):
#     if dataset == '':
#        df_train = pd.read_csv(os.path.join(folder, 'train.csv'))
#        df_test = pd.read_csv(os.path.join(folder, 'test.csv'))
#    else:
#        df_train = pd.read_csv(os.path.join(folder, dataset+'_train.csv'))
#        df_test = pd.read_csv(os.path.join(folder, dataset+'_test.csv'))
    
    na_values = -999
    if dataset == '':
        df_train = readDataset(folder, file='train.csv', missing=na_values)
        df_test = readDataset(folder, file='test.csv', missing=na_values)
    else:
        df_train = readDataset(folder, file=dataset+'_train.csv', missing=na_values)
        df_test = readDataset(folder, file=dataset+'_test.csv', missing=na_values)
    
    if 'lat_lon' in features and ('lat' in df_train.columns and 'lon' in df_test.columns):
        df_train['lat_lon'] = geoHasTransform(df_train)
        df_test['lat_lon']  = geoHasTransform(df_test)
        
    return df_train, df_test