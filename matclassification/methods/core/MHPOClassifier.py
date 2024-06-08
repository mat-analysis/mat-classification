# -*- coding: utf-8 -*-
'''
MAT-classification: Analisys and Classification methods for Multiple Aspect Trajectory Data Mining

The present package offers a tool, to support the user in the task of data analysis of multiple aspect trajectories. It integrates into a unique framework for multiple aspects trajectories and in general for multidimensional sequence data mining methods.
Copyright (C) 2022, MIT license (this portion of code is subject to licensing from source project distribution)

Created on Dec, 2021
Copyright (C) 2022, License GPL Version 3 or superior (see LICENSE file)

@author: Tarlis Portela
'''
# --------------------------------------------------------------------------------
from matclassification.methods.core import *
# --------------------------------------------------------------------------------  

# Hiperparameter Optimization Classifier - For Movelet/Features input data
class MHPOClassifier(MClassifier):
    
    def __init__(self, 
                 name='NN',
                 save_results=False,
                 n_jobs=-1,
                 verbose=False,
                 random_state=42,
                 filterwarnings='ignore'):
        
        super().__init__(name=name, n_jobs=n_jobs, verbose=verbose, random_state=random_state, filterwarnings=filterwarnings)
        
        self.save_results = save_results
        
        np.random.seed(seed=random_state)
        random.set_seed(random_state)
    
    def score(self, X_test, y_test, y_pred):
        acc, acc_top5, _f1_macro, _prec_macro, _rec_macro, bal_acc = compute_acc_acc5_f1_prec_rec(y_test, np.array(y_pred), print_metrics=False)
        
        dic_model = {
            'acc': acc,
            'acc_top_K5': acc_top5,
            'balanced_accuracy': bal_acc,
            'precision_macro': _f1_macro,
            'recall_macro': _prec_macro,
            'f1_macro': _rec_macro,
        } 
        
        return pd.DataFrame(dic_model, index=[0])
    
    def create(self, config=None):
        
        # **** Method to overrite ****
        print('\n['+self.name+':] Warning! you must overwrite the create() method.')

        # Example structure:
        if hasattr(self, 'best_config'):
            self.model = None
        else:
            self.model = None
        
        return self.model
    
    
    ## Overwrite train method to do Hiperparameter Optimization:
    def train(self, dir_validation='.'):
        
        # This implementation, trains only one model 
        # (but, you may overwrite the method following this structure or HPSClassifier.train())
        
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

#            validation_report, y_pred = self.model.predict(X_val, y_val)
            validation_report, y_pred = self.predict(X_val, y_val)

            if self.save_results:
                validation_report.to_csv(filename, index=False)

            data.append( validation_report )

#                self.model.free()
        
        self.report = pd.concat(data)
        self.report.reset_index(drop=True, inplace=True)

        self.report.sort_values('acc', ascending=False, inplace=True)
        
        return self.report