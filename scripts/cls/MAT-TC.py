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

from matdata.dataset import readDataset
from matclassification.methods import *

def parse_args():
    parse = argparse.ArgumentParser(
        description="""MAT Trajectory Classification:
    This script runs classifiers for trajectory input.
    
    Available classifiers are:
    - MARC: multiple-aspect trajectory classification via space, time, and semantic embeddings (Petry et. al. - IJGIS 2020)
    - DeepeST: Using deep learning for trajectory classification (Freitas et. al. - ICAART 2021)
    - TRF: Trajectory Random Forrest (Freitas et. al. - ICAART 2021, from Breiman 2001)
    - TXGB: Trajectory XGBoost (Freitas et. al. - ICAART 2021, from Chen and Guestrin
2016)
    - Tulvae: (Freitas et. al. - ICAART 2021, from Gao et al., 2017)
    - Bituler: (Freitas et. al. - ICAART 2021, from Gao et al., 2017)
    
    Example Usage:
        MAT-TC.py 'sample/data/FoursquareNYC' 'sample/results' -c 'MARC,DeepeST'
        MAT-TC.py 'sample/data/FoursquareNYC' 'sample/results' -c 'Bituler,Tulvae' --one-feature 'poi'
    """, formatter_class=argparse.RawTextHelpFormatter)
    parse.add_argument('data-path', type=str, help='path for the dataset folder')
    parse.add_argument('results-path', type=str, help='path for the results folder')
    parse.add_argument('-pf', '--prefix', type=str, default='', help='dataset name prefix')
    parse.add_argument('-ff', '--file-format', type=str, default='parquet', help='dataset file ext')
    parse.add_argument('-c', '--classifiers', type=str, default='MARC,TRF,TXGB,DeepeST', help='classifiers methods')
    
    parse.add_argument('-r', '--random', type=int, default=1, help='random seed')
    
    parse.add_argument('--geohash', action='store_true', default=False, 
                       help='use GeoHash encoding for spatial aspects (not implemented)')   
    parse.add_argument('-g', '--geo-precision', type=int, default=30, help='Space precision for GeoHash/GridIndex encoding') 
    
    parse.add_argument('-of', '--one-feature', type=str, default='poi', help='[Bituler,Tulvae] Single feature classification (sets attribute name)')

    args = parse.parse_args()
    config = vars(args)
    return config
 
config = parse_args()
#print(config)
    
data_path = config["data-path"]
res_path  = config["results-path"]
prefix    = config["prefix"]
fformat   = config["file_format"]

random_seed   = config["random"]
geohash       = config["geohash"]
geo_precision = config["geo_precision"]

one_feature   = config["one_feature"]

classifiers   = config["classifiers"].split(',')

# ------------------------------------------------------------------------------------
def callmodel(method, train, test):    
    model = eval(method)(random_state=random_seed)
    
     # TODO Grid encoding or geohash + testing
    if isinstance(model, Bituler) or isinstance(model, Tulvae):
        model.prepare_input(train, test, 
                            space_geohash=geohash, 
                            geo_precision=geo_precision, 
                            features=[one_feature])
    elif isinstance(model, MARC):
        model.prepare_input(train, test,
                            geo_precision=geo_precision)
    else:
        model.prepare_input(train, test, 
                            space_geohash=geohash, 
                            geo_precision=geo_precision)
    
    model.train()
    model.test()
    ## We can visualize the training report (the same on most models):
    print(model.summary())
    
    modelfolder = method
    if prefix and prefix != '':
        modelfolder = method+'-'+prefix
    
    # Saving results
    model.save(res_path, modelfolder)
# ------------------------------------------------------------------------------------
train = 'train.'+fformat
test = 'test.'+fformat
if prefix and prefix != '':
    train = prefix + '_' + train
    test = prefix + '_' + test

train = readDataset(os.path.join(data_path, train))
test = readDataset(os.path.join(data_path,  test))

for method in classifiers:
    callmodel(method, train, test)