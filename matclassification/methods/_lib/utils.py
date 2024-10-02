# -*- coding: utf-8 -*-
'''
MAT-Tools: Python Framework for Multiple Aspect Trajectory Data Mining

The present package offers a tool, to support the user in the task of data analysis of multiple aspect trajectories. It integrates into a unique framework for multiple aspects trajectories and in general for multidimensional sequence data mining methods.
Copyright (C) 2022, MIT license (this portion of code is subject to licensing from source project distribution)

Created on Dec, 2021
Copyright (C) 2022, License GPL Version 3 or superior (see LICENSE file)

Authors:
    - Tarlis Portela
'''
# --------------------------------------------------------------------------------
def update_report(df, names, *params):
    """
    Updates a DataFrame with new values for specified columns.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to update.

    names : str
        A comma-separated string of column names to update.

    params : tuple
        A variable number of values to assign to the specified columns.

    Returns:
    --------
    pandas.DataFrame
        The updated DataFrame with new values for the specified columns.
    """
    for index, (att, val) in enumerate(zip(names.split(', '), params)):
        df[att] = [val]
    return df

def print_params(names, *params):
     """
    Formats parameters into a single string, with each parameter
    paired with its corresponding name.

    Parameters:
    -----------
    names : str
        A comma-separated string of parameter names.

    params : tuple
        A variable number of parameter values to format.

    Returns:
    --------
    str
        A string representing the formatted parameters, 
        with each name-value pair joined by an underscore.
    """
    return '_'.join([str(x)+'_'+str(y) for i, (x,y) in enumerate(zip(names.split(', '), params))])

def concat_params(*params):
    """
    Concatenates a variable number of parameters into a single string.

    Parameters:
    -----------
    params : tuple
        A variable number of parameter values to concatenate.

    Returns:
    --------
    str
        A single string with all parameters concatenated,
        separated by a hyphen.
    """
    return '-'.join([str(y) for y in params])