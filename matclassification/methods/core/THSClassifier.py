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

# Hiperparameter Optimization Classifier - For Trajectory input data
class THSClassifier(HSClassifier):
    
    def __init__(self, 
                 name='NN',
                 
                 save_results=False,
                 n_jobs=-1,
                 verbose=False,
                 random_state=42,
                 filterwarnings='ignore'):

        super().__init__(name=name, save_results=save_results, n_jobs=n_jobs, verbose=verbose, random_state=random_state, filterwarnings=filterwarnings)
        
    def message(self, pbar, text):
        if self.isverbose:
            pbar.set_postfix_str(text)
                
    def read_report(self, filename, prefix=''):
        marksplit = '-'
        validation_report = pd.read_csv(filename)
        filename = filename.split(prefix+self.name.lower()+'-')[-1]
        filename = filename.split('.csv')[0]

        i = 0
        for y in filename.split(marksplit):
            validation_report['p'+str(i)] = y
            i+=1
        return validation_report
    
    def xy(self,
           train, test,
           tid_col='tid', 
           class_col='label',
           space_geohash=False, # True: Geohash, False: indexgrid
           geo_precision=30,    # Geohash: precision OR IndexGrid: meters
           validate=False):
        
        # RETURN: X, y, features, num_classes, space, dic_parameters
        return prepareTrajectories(train.copy(), test.copy(),
                                   tid_col=tid_col, 
                                   class_col=class_col,
                                   # space_geohash, True: Geohash, False: indexgrid
                                   space_geohash=space_geohash, 
                                   # Geohash: precision OR IndexGrid: meters
                                   geo_precision=geo_precision,     

                                   features_encoding=True, 
                                   y_one_hot_encodding=False,
                                   split_test_validation=validate,
                                   data_preparation=1,

                                   verbose=self.isverbose)
    
    def prepare_input(self,
                      train, test,
                      tid_col='tid', 
                      class_col='label',
                      space_geohash=False, # True: Geohash, False: indexgrid
                      geo_precision=30,    # Geohash: precision OR IndexGrid: meters
                      validate=False):
        
        # Load Data - Tarlis:
        X, y, features, num_classes, space, dic_parameters = self.xy(train, test, tid_col, class_col, geo_precision, validate)
        
        self.add_config(features=features,
                        num_classes=num_classes, 
                        space=space,
                        dic_parameters=dic_parameters)
        
        if 'encode_y' in dic_parameters.keys():
            self.le = dic_parameters['encode_y']
        
        if len(X) == 2:
            self.X_train = X[0] 
            self.X_test = X[1]
            self.y_train = y[0] 
            self.y_test = y[1]           
            self.validate = False
        if len(X) > 2:
            self.X_train = X[0] 
            self.X_val = X[1]
            self.X_test = X[2]
            self.y_train = y[0] 
            self.y_val = y[1]
            self.y_test = y[2]
            self.validate = True
            
        return X, y, features, num_classes, space, dic_parameters
    
    def fit(self, 
            X_train, 
            y_train, 
            X_val,
            y_val,
            config=None):
                  
        if not config:
            config = self.best_config            
        if not self.model:
            self.model = self.create(config)
        
        return self.model.fit(X_train, 
                              y_train, 
                              X_val,
                              y_val)
    
    def predict(self,                 
                X_test,
                y_test):
        
        y_pred_prob = self.model.predict(X_test) 
        if y_pred_prob.ndim == 1:
            y_pred = y_pred_prob
        else:
            y_pred = argmax(y_pred_prob , axis = 1)

        self.y_test_true = y_test
        self.y_test_pred = y_pred
        
        if self.le:
            self.y_test_true = self.le.inverse_transform(self.y_test_true)
            self.y_test_pred = self.le.inverse_transform(self.y_test_pred)
            
        self._summary = self.score(y_test, y_pred_prob)
        return self._summary, y_pred_prob 
    
    def train(self, dir_validation='.'):
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
        
        # Hiper-param data:
        data = []
        
        self.best_config = [-1, []]
        
        if self.isverbose:
            pbar = tqdm(self.grid, desc='['+self.name+':] Model Training')
        else:
            pbar = self.grid
        
        for config in pbar:
            
            # concat config:
            params = '-'.join([str(y) for y in config])
            filename = os.path.join(dir_validation, self.name.lower()+'-'+params+'.csv')
            
            if os.path.exists(filename):
                self.message(pbar, 'Skip ---> {}'.format(filename))
                data.append(self.read_report(filename))
                
            else:
                self.message(pbar, 'Trainning Config - '+params)
                
                self.model = self.create(config)
                self.fit(X_train, y_train, X_val, y_val, config)

                validation_report, y_pred = self.predict(X_val, y_val)

                if self.save_results:
                    validation_report.to_csv(filename, index=False)

                for index, (att, val) in enumerate(zip(['p'+str(y) for y in range(len(config))], config)):
                    validation_report[att] = [val]

                data.append( validation_report )
                
            # Choose the best model based on higher acc:
            acc = validation_report.iloc[0]['acc']
            if acc > self.best_config[0]:
                self.best_config = [acc, config]
                self.best_model = self.model
            
            self.clear()
            # ------------------------------------------->
            break 
            # TODO -----------------------------------------># -------------------------------------------> **************************

        self.best_config = self.best_config[1]
        
        self.report = pd.concat(data)
        self.report.reset_index(drop=True, inplace=True)

        self.report.sort_values('acc', ascending=False, inplace=True)
        
        self.model = self.best_model
        
        return self.report
        
    def test(self,
             rounds=1,
             dir_evaluation='.'):
        
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
        
        params = '-'.join([str(y) for y in self.best_config])
        filename = os.path.join(dir_evaluation, 'eval_'+self.name.lower()+'-'+params+'.csv')
        
        if os.path.exists(filename):
            if self.isverbose:
                print('['+self.name+':] Model previoulsy built.')
            return self.read_report(filename, prefix='eval_')
        else:
            if self.isverbose:
                print('['+self.name+':] Creating a model to test set')
                print('['+self.name+':] Evaluation Config - '+params)
            
                pbar = tqdm(range(rounds), desc="Model Testing")
            else:
                pbar = list(range(rounds))
                
            random_state = self.config['random_state']
            
            evaluate_report = []
            for e in pbar:
                re = (random_state+e)
                self.config['random_state'] = re
                
                self.message(pbar, 'Round {} of {} (random {})'.format(e, rounds, re))
                
                self.model = self.create(self.best_config)
                
                self.fit(X_train, y_train, X_val, y_val, self.best_config)
                
                eval_report, y_pred = self.predict(X_test, y_test)
                eval_report['cls_time'] = self.duration()
                
                evaluate_report.append(eval_report)
            
                        
            self.config['random_state'] = random_state
            
            self.test_report = pd.concat(evaluate_report)
            self.test_report.reset_index(drop=True, inplace=True)
            
            if self.isverbose:
                print('['+self.name+':] Processing time: {} milliseconds. Done.'.format(self.duration()))

            return self.test_report, y_pred

    def summary(self):
#        tail = self.report.tail(1)
        tail = self.test_report.mean()
        summ = {
            'acc':               tail['acc'],
            'acc_top_K5':        tail['acc_top_K5'],
            'balanced_accuracy': tail['balanced_accuracy'],
            'precision_macro':   tail['precision_macro'],
            'recall_macro':      tail['recall_macro'],
            'f1_macro':          tail['f1_macro'],
#            'loss':              None, # TODO New metrics
            'cls_time':          self.test_report['cls_time'].max()
        }
        
        self._summary = pd.DataFrame(summ, index=[0])
        return self._summary