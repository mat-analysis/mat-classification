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
import os 
import pandas as pd
from datetime import datetime

from tqdm.auto import tqdm
# --------------------------------------------------------------------------------
from matclassification.methods.core import AbstractClassifier
# -------------------------------------------------------------------------------- 

# Hiperparameter Optimization Classifier - For Trajectory input data
class HSClassifier(AbstractClassifier):
    """
    Hyperparameter Optimization Classifier for Trajectory Input Data.

    This class extends `AbstractClassifier` to include functionality for training and testing 
    machine learning models with hyperparameter optimization. It is designed for trajectory 
    data inputs and handles multiple configurations to find the best-performing model.
    
    Check: help(AbstractClassifier)
    
    Parameters:
    -----------
    name : str, optional
        Classifier name (default: 'NN').
    save_results : bool, optional
        Flag to enable saving results to disk (default: False).
    n_jobs : int, optional
        Number of parallel jobs to run (default: -1 for using all processors).
    verbose : bool, optional
        Flag for verbosity (default: False).
    random_state : int, optional
        Random seed for reproducibility (default: 42).
    filterwarnings : str, optional
        Warning filter level (default: 'ignore').

    Methods:
    --------
    train(dir_validation='.'):
        Trains the model using a single hyperparameter configuration.
        If validation is enabled, it will evaluate on a validation set; otherwise, 
        it evaluates on the test set. Results are optionally saved to a CSV file.
        
        Parameters:
        -----------
        dir_validation : str, optional
            Directory where validation results will be saved (default: current directory).
        
        Returns:
        --------
        pd.DataFrame
            A DataFrame containing the training report with evaluation metrics for the model.

    test(rounds=1, dir_evaluation='.'):
        Tests the model over a specified number of rounds, each with a different random seed, 
        to simulate multiple model evaluations.
        
        Parameters:
        -----------
        rounds : int, optional
            The number of evaluation rounds (default: 1).
        dir_evaluation : str, optional
            Directory where evaluation results will be saved (default: current directory).
        
        Returns:
        --------
        pd.DataFrame, np.array
            A DataFrame containing the evaluation report and the predicted labels for the test data.
        
    """
    
    def __init__(self, 
                 name='NN',
                 
                 save_results=False,
                 n_jobs=-1,
                 verbose=False,
                 random_state=42,
                 filterwarnings='ignore'):
        
        super().__init__(name=name, n_jobs=n_jobs, verbose=verbose, random_state=random_state, filterwarnings=filterwarnings)
        
        self.save_results = save_results
        
    ## Overwrite train and test methods to do enable Hiperparameter Optimizations:
    ## (in this case, trains only one default configuration.
    def train(self, dir_validation='.'):
        """
        Trains the model using all hyperparameter configurations.

        If validation is enabled, it will evaluate on a validation set; otherwise, 
        it evaluates on the test set. Results are optionally saved to a CSV file.

        Parameters:
        -----------
        dir_validation : str, optional
            Directory where validation results will be saved (default: current directory).

        Returns:
        --------
        pd.DataFrame
            A DataFrame containing the training report with evaluation metrics for the model.
        """
        # This implementation, trains only one model 
        # (but, you may overwrite the method following this method)
        
        self.start_time = datetime.now()
        
        X_train = self.X_train
        y_train = self.y_train
        
        if self.validate:
            X_val = self.X_val
            y_val = self.y_val
        else:
            X_val = self.X_test
            y_val = self.y_test            
        
        if self.isverbose:
            print('['+self.name+':] Training hiperparameter model')
        
        data = []
        
        # TODO: Hiperparam config training...
        ## This part you may want to run for each configuration (as a progress bar):
        #for config in pbar:
        filename = os.path.join(dir_validation, 'val_'+self.name.lower()+'.csv')
            
        if os.path.exists(filename):
            print('Skip ---> {}'.format(filename))
        else:
            self.model = self.create() # pass the config dict()
            self.fit(X_train, y_train, X_val, y_val) #, config)

            validation_report, y_pred = self.predict(X_val, y_val)
            validation_report['clstime'] = self.duration()

            if self.save_results:
                validation_report.to_csv(filename, index=False)

            data.append( validation_report )

#                self.model.free()
        
        self.report = pd.concat(data)
        self.report.reset_index(drop=True, inplace=True)

        # Use sorting if each train is a different model for hiperparam search, and you are loonig for the best model acc.
        #self.report.sort_values('acc', ascending=False, inplace=True)
        
        return self.report
    
    def test(self,
             rounds=1,
             dir_evaluation='.'):
        """
        Tests the model in teh simgle best model trained, over a specified number of rounds, 
        each with a different random seeds, to simulate multiple model evaluations.

        Parameters:
        -----------
        rounds : int, optional
            The number of evaluation rounds (default: 1).
        dir_evaluation : str, optional
            Directory where evaluation results will be saved (default: current directory).

        Returns:
        --------
        pd.DataFrame, np.array
            A DataFrame containing the evaluation report and the predicted labels for the test data.
        """
        
        X_train = self.X_train
        y_train = self.y_train
        
        if self.validate:
            X_val = self.X_val
            y_val = self.y_val
        else:
            X_val = self.X_test
            y_val = self.y_test  
            
        X_test = self.X_test
        y_test = self.y_test
        
        filename = os.path.join(dir_evaluation, 'eval_'+self.name.lower()+'.csv')
        
        if os.path.exists(filename):
            if self.isverbose:
                print('['+self.name+':] Model previoulsy built.')
            # TODO read
            #return self.read_report(filename, prefix='eval_')
        else:
            if self.isverbose:
                print('['+self.name+':] Creating a model to test set')
            
                pbar = tqdm(range(rounds), desc="Model Testing")
            else:
                pbar = list(range(rounds))
                
            random_state = self.config['random_state']
            
            evaluate_report = []
            for e in pbar:
                re = (random_state+e)
                self.config['random_state'] = re
                
                self.message(pbar, 'Round {} of {} (random {})'.format(e, rounds, re))
                
                self.model = self.create()
                
                self.fit(X_train, y_train, X_val, y_val)
                
                eval_report, y_pred = self.predict(X_test, y_test)
                eval_report['clstime'] = self.duration()
                
                evaluate_report.append(eval_report)
                        
            self.config['random_state'] = random_state
            self.test_report = pd.concat(evaluate_report)
            self.test_report.reset_index(drop=True, inplace=True)
            
            if self.isverbose:
                print('['+self.name+':] Processing time: {} milliseconds. Done.'.format(self.duration()))

            return self.test_report, y_pred