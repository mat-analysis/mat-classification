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

import argparse
from datetime import datetime

from matdata.dataset import readDataset
from matclassification.methods import POIS

def parse_args():
    parse = argparse.ArgumentParser(
        description="""POIS Trajectory Classification:
    This script runs feature extraction and the classifier for POI-S method.
    
    Example Usage:
        POIS-TC.py 'sample/data/FoursquareNYC' 'sample/results'
    """, formatter_class=argparse.RawTextHelpFormatter)
    parse.add_argument('data-path', type=str, help='path for the dataset folder')
    parse.add_argument('results-path', type=str, help='path for saving the POIS result files')
    parse.add_argument('-m', '--method', type=str, default='npoi', help='POIF method name (poi, [npoi], wnpoi)')
    parse.add_argument('-s', '--sequences', type=str, default='1,2,3', help='sequences sizes to concatenate')
    parse.add_argument('-f', '--features', type=str, default='poi', help='feature names to concatenate (attributes)')
    
    parse.add_argument('-pf', '--prefix', type=str, default=None, help='dataset name prefix')
    parse.add_argument('-ff', '--file-format', type=str, default='parquet', help='dataset file ext')
    
    parse.add_argument('-r', '--random', type=int, default=1, help='random seed')
    
    parse.add_argument('--geohash', action='store_true', default=False, 
                       help='use GeoHash encoding for spatial aspects (not implemented)')   
    parse.add_argument('-g', '--geo-precision', type=int, default=30, help='Space precision for GeoHash/GridIndex encoding') 
    
    parse.add_argument('--save-extraction', action='store_true', default=False, help='Save feature extraction files')

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

random_seed = config["random"]
save        = config["save_extraction"]

# TODO:
geohash       = config["geohash"]
geo_precision = config["geo_precision"]


core_name = '_'.join(features) + '_' + '_'.join([str(s) for s in sequences])
folder = method.upper()+'-'+core_name

# Starting:
# ---------------------------------------------------------------------------------
time = datetime.now()
# Input:
train = 'train.'+fformat
test = 'test.'+fformat
if prefix and prefix != '':
    train = prefix + '_' + train
    test = prefix + '_' + test

train = readDataset(os.path.join(data_path, train))
test = readDataset(os.path.join(data_path,  test))

# Classification:
model = POIS(method, sequences, features)
model.prepare_input(train, test, res_path=os.path.join(res_path, folder) if save else None)
model.train()
model.test()
    
print(model.summary())

model.save(os.path.join(res_path, folder), core_name)
# ---------------------------------------------------------------------------------
time_ext = (datetime.now()-time).total_seconds() * 1000

print("Done. Processing time: " + str(time_ext) + " milliseconds")
print("# ---------------------------------------------------------------------------------")