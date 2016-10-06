import numpy as np

from keras.models import Sequential
from keras.layers import Activation
from keras.layers.convolutional import Convolution1D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adadelta

class AllCNN():
    def __init__(self, n_filters = 3, filter_length = 3, stride = 5, n_elements = 240000, n_ch = 16, n_conv_per_layer = 2, n_layers = 5):
        self.n_filters = n_filters
        self.filter_length = filter_length
        self.stride = stride
        self.n_elements = n_elements
        self.n_ch = n_ch
        self.n_conv_per_layer = n_conv_per_layer
        self.n_layers = n_layers
        
    def build_model(self):
        init = 'he_normal'
        
        model = Sequential()
        model.add(BatchNormalization(input_shape = (self.n_elements, self.n_ch), axis = 1))
        model.add(Convolution1D(self.n_filters, self.filter_length, border_mode = 'same', activation = 'relu', init = init))
        for i in range(2, self.n_conv_per_layer):
            model.add(Convolution1D(self.n_filters, self.filter_length, border_mode = 'same', activation = 'relu', init = init))
        model.add(Convolution1D(self.n_filters, self.filter_length, border_mode = 'same', activation = 'relu', subsample_length = self.stride, init = init))
        
        for i in range(1, self.n_layers):
            for j in range(1, self.n_conv_per_layer):
                model.add(Convolution1D(self.n_filters, self.filter_length, border_mode = 'same', activation = 'relu', init = init))
            model.add(Convolution1D(self.n_filters, self.filter_length, border_mode = 'same', activation = 'relu', subsample_length = self.stride, init = init))
            
        model.add(Flatten())
        
        model.add(Dense(model.output_shape[1], init = init))
        model.add(Dropout(.5))
        model.add(Activation('relu'))
        
        model.add(Dense(1, init = init))
        model.add(Activation('sigmoid'))
        
        model.compile(optimizer = 'adadelta',
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])
        
        return model
        
    def train(self, model, eeg_reader, callbacks, n_epochs = 100, samples_per_epoch = 500):
        model.fit_generator(eeg_reader.gen_train(), samples_per_epoch, n_epochs, validation_data = eeg_reader.gen_val(), nb_val_samples = eeg_reader.n_val_samples, class_weight = eeg_reader.class_weights, callbacks = callbacks)
    
    def predict(self, model, eeg_reader):
        name_arr = np.array([''] * eeg_reader.n_samples, dtype = object)
        prediction_arr = np.zeros(eeg_reader.n_samples)
        
        idx = 0
        for names, test_x in eeg_reader.gen_test():
            print(' %d/%d' % (idx, eeg_reader.n_samples))
            batch_samples = len(names)
            prediction = model.predict_proba(test_x, batch_size = batch_samples, verbose = 0)
            prediction_arr[idx:idx + batch_samples] = prediction.flatten()
            name_arr[idx:idx + batch_samples] = names
            
            idx = idx + batch_samples
            
        return name_arr, prediction_arr
            