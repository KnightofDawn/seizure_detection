from os import mkdir
from os.path import isfile, isdir

import numpy as np

from keras.models import Sequential, load_model
from keras.layers import Activation
from keras.layers.convolutional import Convolution1D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.callbacks import Callback, ModelCheckpoint
from keras import backend as K

from sklearn.metrics import roc_auc_score, classification_report

from eeg_generator import EEGGenerator

def build_cnn(out_file):
    filters = 3
    filter_length = 3
    stride = 5
    n_elements = 240000
    n_ch = 16
    
    init = 'he_normal'
    
    model = Sequential()
    model.add(Convolution1D(filters, filter_length, border_mode = 'same', input_shape = (n_elements, n_ch), activation = 'relu', init = init))
    model.add(Convolution1D(filters, filter_length, border_mode = 'same', activation = 'relu', subsample_length = stride, init = init))
    
    model.add(Convolution1D(filters, filter_length, border_mode = 'same', activation = 'relu', init = init))
    model.add(Convolution1D(filters, filter_length, border_mode = 'same', activation = 'relu', subsample_length = stride, init = init))
    
    model.add(Convolution1D(filters, filter_length, border_mode = 'same', activation = 'relu', init = init))
    model.add(Convolution1D(filters, filter_length, border_mode = 'same', activation = 'relu', subsample_length = stride, init = init))
    
    model.add(Convolution1D(filters, filter_length, border_mode = 'same', activation = 'relu', init = init))
    model.add(Convolution1D(filters, filter_length, border_mode = 'same', activation = 'relu', subsample_length = stride, init = init))
    
    model.add(Convolution1D(filters, filter_length, border_mode = 'same', activation = 'relu', init = init))
    model.add(Convolution1D(filters, filter_length, border_mode = 'same', activation = 'relu', subsample_length = stride, init = init))
    
    model.add(Flatten())
    
    model.add(Dense(128, init = init))
    model.add(Dropout(.5))
    model.add(Activation('relu'))
    
    model.add(Dense(1, init = init))
    model.add(Activation('sigmoid'))
    
    print(model.summary())
    
    model.compile(optimizer='adadelta',
              loss='binary_crossentropy',
              metrics=['accuracy'])
    
    model.save(out_file)
    
    del model
    
class Validation_ROC_AUC(Callback):
    def __init__(self, gen, n_batches = 16):
        self.gen = gen
        self.n_batches = n_batches
        
    def on_epoch_end(self, epoch, logs={}):
        print('')
        y_true = np.array([])
        y_pred = np.array([])
        for i in range(self.n_batches):
            x, y = next(self.gen.gen_train())
            pred = self.model.predict_proba(x, verbose = 0)[:, 0]
            y_true = np.append(y_true, y)
            y_pred = np.append(y_pred, pred)
        print('ROC_AUC: %.3f' % roc_auc_score(y_true, y_pred))

def main():
    model_dir = 'data/models/allcnn1'
    model_file = 'data/models/allcnn1/cnn.h5'
    best_weights_file = 'data/models/allcnn1/best_weights.h5'
    epoch_weights_file = 'data/models/allcnn1/loss_{loss:.3f}_weights.h5'
    if not isdir(model_dir):
        mkdir(model_dir)
    if not isfile(model_file):
        build_cnn(model_file)
    
    model = load_model(model_file)
    
    if isfile(best_weights_file):
        model.load_weights(best_weights_file)
    
    
    with EEGGenerator('data/train.h5') as gen:
        print(gen.class_weights)
        
        epoch_cp = ModelCheckpoint(epoch_weights_file, monitor = 'val_loss', save_weights_only = True)
        best_cp = ModelCheckpoint(best_weights_file, monitor = 'val_loss', save_weights_only = True, save_best_only = True)
        
        samples_per_epoch = gen.batch_size * 64
        n_epochs = 100
        
        model.fit_generator(gen.gen_train(), samples_per_epoch, n_epochs, validation_data = gen.gen_val(), nb_val_samples = gen.n_val_samples, class_weight = gen.class_weights, callbacks = [epoch_cp, best_cp])
    
if __name__ == '__main__':
    main()       
                
            