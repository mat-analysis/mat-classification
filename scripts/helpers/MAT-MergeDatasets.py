#!python
# -*- coding: utf-8 -*-
'''
Multiple Aspect Trajectory Data Mining Tool Library

The present application offers a tool, to support the user in the classification task of multiple aspect trajectories, specifically for extracting and visualizing the movelets, the parts of the trajectory that better discriminate a class. It integrates into a unique platform the fragmented approaches available for multiple aspects trajectories and in general for multidimensional sequence classification into a unique web-based and python library system. Offers both movelets visualization and a complete configuration of classification experimental settings.

Created on Dec, 2021
Copyright (C) 2022, License GPL Version 3 or superior (see LICENSE file)

@author: Tarlis Portela
'''
import os
import pandas as pd
import numpy as np
import glob2 as glob

import argparse

def parse_args():
    """[This is a function used to parse command line arguments]

    Returns:
        args ([object]): [Parse parameter object to get parse object]
    """
    parse = argparse.ArgumentParser(description='Merge train.csv/test.csv class files for classifier input')
    parse.add_argument('results-path', type=str, help='path for the results folder')

    args = parse.parse_args()
    config = vars(args)
    return config
 
config = parse_args()
#print(config)

results_path    = config["results-path"]

def mergeDatasets(dir_path, file='train.csv'):
#     from ..main import importer
    
#    files = [i for i in glob.glob(os.path.join(dir_path, '*', '**', file))]
    files = list(map(lambda f: f, glob.glob(os.path.join(dir_path, '*', '**', file))))
    by_time = {os.path.getctime(file): file for file in files}
    keys = sorted(by_time.keys())
    
    files = [by_time[i] for i in keys]
    
    print("Loading files - " + file)
    # combine all files in the list
#    combined_csv = pd.concat([pd.read_csv(f).drop(['tid','label'], axis=1, errors='ignore') for f in files[:len(files)-1]], axis=1)
    combined_csv = list(map(lambda f: pd.read_csv(f).drop(['tid','class','label'], axis=1, errors='ignore'), files[:len(files)-1] ))
    combined_csv = pd.concat(combined_csv + [pd.read_csv(files[len(files)-1])], axis=1)
    
    combined_csv.rename(columns={'class': 'label'}, errors="ignore", inplace=True)
    
    #export to csv
    print("Writing "+file+" file")
    combined_csv.to_csv(os.path.join(dir_path, file), index=False)
    
    print("Done.")

mergeDatasets(results_path, 'train.csv')
mergeDatasets(results_path, 'test.csv')