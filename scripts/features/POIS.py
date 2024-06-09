#!python
# -*- coding: utf-8 -*-
'''
Multiple Aspect Trajectory Data Mining Tool Library

The present application offers a tool, to support the user in the classification task of multiple aspect trajectories, specifically for extracting and visualizing the movelets, the parts of the trajectory that better discriminate a class. It integrates into a unique platform the fragmented approaches available for multiple aspects trajectories and in general for multidimensional sequence classification into a unique web-based and python library system. Offers both movelets visualization and a complete configuration of classification experimental settings.

Created on Dec, 2021
Copyright (C) 2022, License GPL Version 3 or superior (see LICENSE file)

@author: Tarlis Portela
'''
import sys, os 
sys.path.insert(0, os.path.abspath('.'))

from datetime import datetime
import argparse

from matdata.dataset import readDataset

from matclassification.methods.feature.feature_extraction.pois import pois
from matclassification.methods.feature.POIS import POIS, prepareData, loadData

def parse_args():
    parse = argparse.ArgumentParser(
        description="""POIS Feature Extraction:
    This script runs feature extraction for POI-S method.
    (Also can run the classifier)
    
    Example Usage:
        POIS.py 'sample/data/FoursquareNYC' 'sample/results'
        POIS.py 'sample/data/FoursquareNYC' 'sample/results' --classify
    """, formatter_class=argparse.RawTextHelpFormatter)
    """[This is a function used to parse command line arguments]

    Returns:
        args ([object]): [Parse parameter object to get parse object]
    """
    parse.add_argument('data-path', type=str, help='path for the dataset folder')
    parse.add_argument('results-path', type=str, help='path for saving the POIS result files')
    parse.add_argument('-m', '--method', type=str, default='npoi', help='POIF method name (poi, [npoi], wnpoi)')
    parse.add_argument('-s', '--sequences', type=str, default='1,2,3', help='sequences sizes to concatenate')
    parse.add_argument('-f', '--features', type=str, default='poi', help='feature names to concatenate (attributes)')
    
    parse.add_argument('-pf', '--prefix', type=str, default=None, help='dataset name prefix')
    parse.add_argument('-ff', '--file-format', type=str, default='parquet', help='dataset file ext')
    
#    parse.add_argument('-f', '--result-folder', type=str, default='npoi-poi-1_2_3', help='folder where to find the POIS processed files')    
    parse.add_argument('-r', '--random', type=int, default=1, help='random seed')
    
    parse.add_argument('--geohash', action='store_true', default=False, 
                       help='use GeoHash encoding for spatial aspects (not implemented)')   
    parse.add_argument('-g', '--geo-precision', type=int, default=30, help='Space precision for GeoHash/GridIndex encoding') 
    
    parse.add_argument('--classify', action='store_true', default=False, help='Do also classification?')
    
    args = parse.parse_args()
    config = vars(args)
    return config
 
config = parse_args()
#print(config)

data_path = config["data-path"]
res_path  = config["results-path"]
prefix    = config["prefix"]
fformat   = config["file_format"]

method      = config["method"]
sequences   = [int(x) for x in config["sequences"].split(',')]
features    = config["features"].split(',')

random_seed   = config["random"]
classify      = config["classify"]

# TODO:
geohash       = config["geohash"]
geo_precision = config["geo_precision"]

# Starting:
# ---------------------------------------------------------------------------------
time = datetime.now()

#sequences = [1,2,3]
#features = ['poi']
#method='npoi' # default: 'npoi', or, 'poi' and 'wnpoi'

# Input:
train = 'train.'+fformat
test = 'test.'+fformat
if prefix and prefix != '':
    train = prefix + '_' + train
    test = prefix + '_' + test

import pandas as pd
train = readDataset(os.path.join(data_path, train))
test = readDataset(os.path.join(data_path,  test))

# Feature extraction:
core_name = '_'.join(features) + '_' + '_'.join([str(s) for s in sequences])
folder = method.upper()+'-'+core_name
res_dir = os.path.join(res_path, folder)
x_train, x_test, y_train, y_test, _ = pois(train, test, sequences, features, method, result_dir=res_dir, save_all=True)
# ---------------------------------------------------------------------------------
def callmodel(res_dir, core_name):
    time_cls = datetime.now()
    
    # POIS method have a method for reading and data transformation:
    x_train, x_test, y_train, y_test = loadData(os.path.join(res_dir, core_name))
    num_features, num_classes, labels, X, y, one_hot_y = prepareData(x_train, x_test, y_train, y_test)
    x_train, x_test = X
    y_train, y_test = y

    # Create the classifier:
    model = POIS(method, sequences, features)

    # Model Label Encoder:
    model.le = one_hot_y
    
    # You can add method variables with this:
    model.add_config(num_features=num_features,
                    num_classes=num_classes, 
                    labels=labels)

    # Run the classifier:
    model.fit(x_train, y_train, x_test, y_test)

    summary, _ = model.predict(x_test, y_test)
    summary['cls_time'] = (datetime.now()-time_cls).total_seconds() * 1000
    print(summary)   
    
    model.test_report = summary
    
    # Saving results
    model.save(res_dir, core_name)
    del model
# ---------------------------------------------------------------------------------
if classify:
    callmodel(res_dir, method.lower()+'_'+core_name)
    
    # For other sequence sizes:
    core_name = method.lower()+'_'+ '_'.join(features) + '_'
    for s in sequences:
        callmodel(res_dir, core_name+str(s))
# ---------------------------------------------------------------------------------

time_ext = (datetime.now()-time).total_seconds() * 1000
print("Done. Processing time: " + str(time_ext) + " milliseconds")
print("# ---------------------------------------------------------------------------------")