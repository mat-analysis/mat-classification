import time
import pandas as pd
import numpy as np

from tqdm.auto import tqdm

# --------------------------------------------------------------------------------
from sklearn import preprocessing
# --------------------------------------------------------------------------------
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.utils import to_categorical
from matanalysis.methods._lib.pymove.models import metrics
from matanalysis.methods._lib.metrics import f1

from matanalysis.methods.models import ModelClassifier

class MoveletMLP(ModelClassifier):
    
    def __init__(self, 
                 par_droupout = 0.5,
                 par_batch_size = 200,
                 par_epochs = 80,
                 par_lr = 0.00095,
                 lst_par_epochs = [80,50,50,30,20],
                 lst_par_lr = [0.00095,0.00075,0.00055,0.00025,0.00015],
                 nattr=-1,
                 nclasses=-1,
                 n_jobs=-1,
                 verbose=False,
                 random_state=42):
        super(self, 'MLP', n_jobs, verbose, random_state)
        
        self.add_config(par_droupout, par_batch_size, par_epochs, par_lr, lst_par_epochs, lst_par_lr, nattr, nclasses)
        
        #Initializing Neural Network
        self.classifier = Sequential()
        # Adding the input layer and the first hidden layer
        self.classifier.add(Dense(units = 100, kernel_initializer = 'uniform', kernel_regularizer= regularizers.l2(0.02), activation = 'relu', input_dim = (nattr)))
        #classifier.add(BatchNormalization())
        self.classifier.add(Dropout( par_dropout )) 
        # Adding the output layer       
        self.classifier.add(Dense(units = nclasses, kernel_initializer = 'uniform', activation = 'softmax'))
        # Compiling Neural Network
    #     adam = Adam(lr=par_lr) # TODO: check for old versions...
        adam = Adam(learning_rate=par_lr)
        self.classifier.compile(optimizer = adam, loss = 'categorical_crossentropy', metrics = ['accuracy','top_k_categorical_accuracy',f1])
        
        
        
        #assert (eval_metric == 'merror') | (eval_metric == 'mlogloss'), "ERR: invalid loss, set loss as mlogloss or merror" 

        #print('[MODEL:] Starting model training, {} iterations'.format(total))

        
        print('['+self.name+':] Processing time: {} milliseconds. Done.'.format(self.duration()))
        
        def fit(self, 
                X_train, 
                y_train, 
                X_val,
                y_val):
            
            # Scaling y and transforming to keras format
            self.le = preprocessing.LabelEncoder()
            le.fit(y_train)
            
            y_train = self.le.transform(y_train) 
            y_test = self.le.transform(y_test)
            
            y_train1 = to_categorical(y_train)
            y_test1 = to_categorical(y_test)
            
            par_batch_size = self.config['par_batch_size']
            par_epochs = self.config['par_epochs']
            verbose=self.config['verbose']
            
            return self.classifier.fit(X_train, y_train1, validation_data = (X_test, y_test1), batch_size = par_batch_size, epochs=par_epochs, verbose=verbose)