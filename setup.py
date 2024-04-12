# -*- coding: utf-8 -*-
'''
MAT-analysis: Analysis and Classification methods for Multiple Aspect Trajectory Data Mining

The present package offers a tool, to support the user in the task of data analysis of multiple aspect trajectories. It integrates into a unique framework for multiple aspects trajectories and in general for multidimensional sequence data mining methods.

Created on Dec, 2021
Copyright (C) 2022, License GPL Version 3 or superior (see LICENSE file)

@author: Tarlis Portela
'''
import setuptools

import configparser
config = configparser.ConfigParser()
config.read('pyproject.toml')
VERSION = config['project']['version'].strip('"')
PACKAGE_NAME = config['project']['name'].strip('"')
DEV_VERSION = "0.1b0"

with open("matclassification/README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
    
setuptools.setup(
    name=PACKAGE_NAME,
    version=VERSION,
#    version=DEV_VERSION,
    author="Tarlis Tortelli Portela",
    author_email="tarlis@tarlis.com.br",
    description="MAT-classification: Analysis and Classification methods for Multiple Aspect Trajectory Data Mining",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ttportela/mat-analysis",
#    packages=setuptools.find_packages(include=[PACKAGE_NAME, PACKAGE_NAME+'.*']),
    packages=setuptools.find_packages(),
#    include_package_data=True,
    scripts=[
        'matanalysis/scripts/helpers/MAT-Summary.py',
        'matanalysis/scripts/helpers/MAT-ResultsTo.py',
        'matanalysis/scripts/helpers/MAT-ExportResults.py',
        'matanalysis/scripts/helpers/MAT-MergeDatasets.py',
        'matanalysis/scripts/helpers/MAT-PrintResults.py',
        'matanalysis/scripts/helpers/MAT-ResultsTo.py',
        
        'matanalysis/scripts/cls/MARC.py',
        'matanalysis/scripts/cls/POIS-TC.py',
        'matanalysis/scripts/cls/MAT-MC.py',
        'matanalysis/scripts/cls/MAT-TC.py',
        'matanalysis/scripts/cls/MAT-TEC.py', # Under Dev.
        
        'matanalysis/scripts/features/POIS.py',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    keywords='data mining, python, trajectory classification, trajectory analysis, movelets',
    license='GPL Version 3 or superior (see LICENSE file)',
)
