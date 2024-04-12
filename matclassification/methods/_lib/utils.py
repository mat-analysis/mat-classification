# -*- coding: utf-8 -*-
'''
MAT-analysis: Analisys and Classification methods for Multiple Aspect Trajectory Data Mining

The present package offers a tool, to support the user in the task of data analysis of multiple aspect trajectories. It integrates into a unique framework for multiple aspects trajectories and in general for multidimensional sequence data mining methods.

Created on Dec, 2021
Copyright (C) 2022, License GPL Version 3 or superior (see LICENSE file)

@author: Tarlis Portela
'''
# --------------------------------------------------------------------------------
def update_report(df, names, *params):
    for index, (att, val) in enumerate(zip(names.split(', '), params)):
        df[att] = [val]
    return df

def print_params(names, *params):
    return '_'.join([str(x)+'_'+str(y) for i, (x,y) in enumerate(zip(names.split(', '), params))])

def concat_params(*params):
    return '-'.join([str(y) for y in params])