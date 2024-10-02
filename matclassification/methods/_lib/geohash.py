# -*- coding: utf-8 -*-
'''
MAT-Tools: Python Framework for Multiple Aspect Trajectory Data Mining

The present package offers a tool, to support the user in the task of data analysis of multiple aspect trajectories. It integrates into a unique framework for multiple aspects trajectories and in general for multidimensional sequence data mining methods.
Copyright (C) 2022, MIT license (this portion of code is subject to licensing from source project distribution)

Created on Dec, 2021
Copyright (C) 2022, License GPL Version 3 or superior (see LICENSE file)

Authors:
    - Tarlis Portela
    - Lucas May Petry (adapted)
'''
# --------------------------------------------------------------------------------
import geohash2 as gh
import numpy as np


base32 = ['0', '1', '2', '3', '4', '5', '6', '7',
          '8', '9', 'b', 'c', 'd', 'e', 'f', 'g',
          'h', 'j', 'k', 'm', 'n', 'p', 'q', 'r',
          's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
binary = [np.asarray(list('{0:05b}'.format(x, 'b')), dtype=int)
          for x in range(0, len(base32))]
base32toBin = dict(zip(base32, binary))


# Deprecated - for compatibility purposes
class LatLonHash:
    """
    Class for encoding latitude and longitude into geohash and binary formats.
    """
    def __init__(self, lat, lon):
        self._lat = lat
        self._lon = lon

    def to_hash(self, precision=15):
        return gh.encode(self._lat, self._lon, precision)

    def to_binary(self, precision=15):
        hashed = self.to_hash(precision)
        return np.concatenate([base32toBin[x] for x in hashed])


def geohash(lat, lon, precision=15):
    """
    Encode latitude and longitude into a geohash.

    Parameters:
    -----------
    lat : float
        Latitude value.
    lon : float
        Longitude value.
    precision : int
        Precision for the geohash (default is 15).

    Returns:
    --------
    str
        The geohash representation of the coordinates.
    """
    return gh.encode(lat, lon, precision)


def bin_geohash(lat, lon, precision=15):
    """
    Encode latitude and longitude into a binary geohash.

    Parameters:
    -----------
    lat : float
        Latitude value.
    lon : float
        Longitude value.
    precision : int
        Precision for the geohash (default is 15).

    Returns:
    --------
    numpy.ndarray
        The binary representation of the geohash.
    """
    hashed = geohash(lat, lon, precision)
    return np.concatenate([base32toBin[x] for x in hashed])
