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
    """
    Reads datasets and applies the POI extraction methods to generate features based on specified sequences.
    (Wrapper for 'pois' method)

    Parameters:
    -----------
    sequences : list of int
        A list of sequence lengths to analyze for POI extraction.

    features : list of str
        The list of features to analyze from the dataset.

    method : str, optional
        The method to use for POI extraction ('poi', 'npoi', or 'wnpoi'). Defaults to 'npoi'.

    dataset : str, optional
        The name of the dataset to process. Defaults to 'specific'.

    folder : str, optional
        The folder path where the dataset files are located. Defaults to './data'.

    result_dir : str, optional
        The directory path where results will be saved. Defaults to '.'.

    save_all : bool, optional
        If True, saves all possible sequences to the result directory. Defaults to False.

    tid_col : str, optional
        The name of the column representing the trajectory ID in the datasets. Defaults to 'tid'.

    class_col : str, optional
        The name of the column representing the class label in the datasets. Defaults to 'label'.

    Returns:
    --------
    tuple
        A tuple containing the aggregated training feature matrix, testing feature matrix, 
        training labels, testing labels, and the core name for the processed data.
    """
    
    df_train, df_test = loadTrainTest(features, folder, dataset)
    
    return pois(df_train, df_test, sequences, features, method, result_dir, save_all, tid_col, class_col)

def pois(df_train, df_test, sequences, features, method='npoi', result_dir='.', save_all=False, tid_col='tid', class_col='label', verbose=True):
    """
    Extracts features from the training and testing datasets based on specified sequences and methods 
    (POI, NPOI, WNPOI) for trajectory classification.

    Parameters:
    -----------
    df_train : pandas.DataFrame
        The training dataset containing trajectory data, including time and location information.

    df_test : pandas.DataFrame
        The testing dataset containing trajectory data for evaluation.

    sequences : list of int
        List of integers specifying the sequence lengths to consider for feature extraction.

    features : list of str
        List of feature names from the datasets to be used for extraction. If None, the function will 
        automatically determine a feature based on variance.

    method : str, optional
        The method to use for feature extraction. Options include:
        - 'poi': Point of Interest frequency.
        - 'npoi': Normalized Point of Interest frequency.
        - 'wnpoi': Weighted Normalized Point of Interest frequency.
        Defaults to 'npoi'.

    result_dir : str, optional
        Directory path where results should be saved. Defaults to the current directory.

    save_all : bool, optional
        If True, all intermediate results will be saved to the specified directory. Defaults to False.

    tid_col : str, optional
        Name of the column representing the trajectory ID in the datasets. Defaults to 'tid'.

    class_col : str, optional
        Name of the column representing the class label in the datasets. Defaults to 'label'.

    verbose : bool, optional
        If True, prints detailed information about the processing steps. Defaults to True.

    Returns:
    --------
    agg_x_train : pandas.DataFrame
        A DataFrame containing aggregated features for the training dataset.

    agg_x_test : pandas.DataFrame
        A DataFrame containing aggregated features for the testing dataset.

    y_train : numpy.ndarray
        A numpy array containing the labels for the training dataset.

    y_test : numpy.ndarray
        A numpy array containing the labels for the testing dataset.

    core_name : str
        A string representing the core name for the generated feature files, based on the selected method 
        and features.
    """
    
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
   
    core_name = method+'_'+('_'.join(features))+'_'+('_'.join([str(n) for n in sequences]))  #+'_'+dataset)
    if save_all:
        to_file(os.path.join(result_dir, core_name), agg_x_train, agg_x_test, y_train, y_test)

    time_ext = (datetime.now()-time).total_seconds() * 1000
    if verbose:
        print('[POIS:] Processing time: {} milliseconds. Done.'.format(time_ext))
        print('------------------------------------------------------------------------------------------------')
    
    return agg_x_train, agg_x_test, y_train, y_test, core_name
    
## --------------------------------------------------------------------------------------------
## POI-F: POI Frequency
def poi(df_train, df_test, possible_sequences, seq2idx, sequence, feature, result_dir=None, tid_col='tid', class_col='label'):
    """
    Computes Point of Interest (POI) frequency features for training and testing datasets.

    Parameters:
    -----------
    df_train : pandas.DataFrame
        The training dataset containing trajectory data.

    df_test : pandas.DataFrame
        The testing dataset containing trajectory data.

    possible_sequences : list of tuple
        List of possible sequences to consider for feature extraction.

    seq2idx : dict
        A dictionary mapping sequences to their corresponding indices.

    sequence : int
        The length of the sequences to be considered.

    feature : str
        The name of the feature to be analyzed in the datasets.

    result_dir : str, optional
        Directory path to save the results. If None, results will not be saved.

    tid_col : str, optional
        The name of the column representing the trajectory ID in the datasets. Defaults to 'tid'.

    class_col : str, optional
        The name of the column representing the class label in the datasets. Defaults to 'label'.

    Returns:
    --------
    x_train : numpy.ndarray
        A 2D array of shape (number of trajectories, number of possible sequences) containing the POI frequencies for the training set.

    x_test : numpy.ndarray
        A 2D array of shape (number of trajectories, number of possible sequences) containing the POI frequencies for the testing set.

    y_train : numpy.ndarray
        A 1D array of class labels for the training dataset.

    y_test : numpy.ndarray
        A 1D array of class labels for the testing dataset.
    """
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
    """
    Computes Normalized Point of Interest (NPOI) frequency features for training and testing datasets.

    Parameters:
    -----------
    df_train : pandas.DataFrame
        The training dataset containing trajectory data.

    df_test : pandas.DataFrame
        The testing dataset containing trajectory data.

    possible_sequences : list of tuple
        List of possible sequences to consider for feature extraction.

    seq2idx : dict
        A dictionary mapping sequences to their corresponding indices.

    sequence : int
        The length of the sequences to be considered.

    feature : str
        The name of the feature to be analyzed in the datasets.

    result_dir : str, optional
        Directory path to save the results. If None, results will not be saved.

    tid_col : str, optional
        The name of the column representing the trajectory ID in the datasets. Defaults to 'tid'.

    class_col : str, optional
        The name of the column representing the class label in the datasets. Defaults to 'label'.

    Returns:
    --------
    x_train : numpy.ndarray
        A 2D array of shape (number of trajectories, number of possible sequences) containing the normalized POI frequencies for the training set.

    x_test : numpy.ndarray
        A 2D array of shape (number of trajectories, number of possible sequences) containing the normalized POI frequencies for the testing set.

    y_train : numpy.ndarray
        A 1D array of class labels for the training dataset.

    y_test : numpy.ndarray
        A 1D array of class labels for the testing dataset.
    """
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
    """
    Computes Weighted Normalized Point of Interest (WNPOI) frequency features for training and testing datasets.

    Parameters:
    -----------
    df_train : pandas.DataFrame
        The training dataset containing trajectory data.

    df_test : pandas.DataFrame
        The testing dataset containing trajectory data.

    possible_sequences : list of tuple
        List of possible sequences to consider for feature extraction.

    seq2idx : dict
        A dictionary mapping sequences to their corresponding indices.

    sequence : int
        The length of the sequences to be considered.

    feature : str
        The name of the feature to be analyzed in the datasets.

    result_dir : str, optional
        Directory path to save the results. If None, results will not be saved.

    tid_col : str, optional
        The name of the column representing the trajectory ID in the datasets. Defaults to 'tid'.

    class_col : str, optional
        The name of the column representing the class label in the datasets. Defaults to 'label'.

    Returns:
    --------
    x_train : numpy.ndarray
        A 2D array of shape (number of trajectories, number of possible sequences) containing the weighted normalized POI frequencies for the training set.

    x_test : numpy.ndarray
        A 2D array of shape (number of trajectories, number of possible sequences) containing the weighted normalized POI frequencies for the testing set.

    y_train : numpy.ndarray
        A 1D array of class labels for the training dataset.

    y_test : numpy.ndarray
        A 1D array of class labels for the testing dataset.
    """
    
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
    """
    Extracts Point of Interest (POI) frequency features for a given dataset and saves the results. 
    For command line use.

    Parameters:
    -----------
    sequence : int
        The length of the sequences to be considered for POI frequency extraction.

    dataset : str
        The name of the dataset to process (without extension).

    feature : str
        The name of the feature to analyze in the dataset.

    folder : str
        The folder path where the dataset files are located.

    result_dir : str
        The directory path where results will be saved.

    tid_col : str, optional
        The name of the column representing the trajectory ID in the datasets. Defaults to 'tid'.

    class_col : str, optional
        The name of the column representing the class label in the datasets. Defaults to 'label'.

    Returns:
    --------
    None
    """
    
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
    """
    Saves the training and testing feature matrices and labels to CSV files.

    Parameters:
    -----------
    core_name : str
        The base name for the output files.

    x_train : numpy.ndarray
        The training feature matrix.

    x_test : numpy.ndarray
        The testing feature matrix.

    y_train : numpy.ndarray
        The training labels.

    y_test : numpy.ndarray
        The testing labels.

    Returns:
    --------
    None
    """
    
    df_x_train = pd.DataFrame(x_train).to_csv(core_name+'-x_train.csv', index=False)#, header=None)
    df_x_test = pd.DataFrame(x_test).to_csv(core_name+'-x_test.csv', index=False)#, header=None)
    df_y_train = pd.DataFrame(y_train, columns=['label']).to_csv(core_name+'-y_train.csv', index=False)
    df_y_test = pd.DataFrame(y_test, columns=['label']).to_csv(core_name+'-y_test.csv', index=False)
    
def geoHasTransform(df, geo_precision=8):
    """
    Transforms latitude and longitude values into geohash representations.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing 'lat' and 'lon' columns.

    geo_precision : int, optional
        The precision for the geohash transformation. Defaults to 8.

    Returns:
    --------
    list
        A list of geohash values corresponding to the latitude and longitude pairs in the DataFrame.
    """
    return [geohash(df['lat'].values[i], df['lon'].values[i], geo_precision) for i in range(0, len(df))]

def loadTrainTest(features, folder, dataset=''):
    """
    Loads the training and testing datasets from CSV files, applying necessary transformations.

    Parameters:
    -----------
    features : list of str
        The features to load from the datasets.

    folder : str
        The folder path where the dataset files are located.

    dataset : str, optional
        The name of the dataset to process (without extension). If empty, default 'train' and 'test' files are loaded.

    Returns:
    --------
    tuple
        A tuple containing the training DataFrame and the testing DataFrame.
    """
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