from os import mkdir
from os.path import isfile, isdir

from keras.models import Sequential, load_model
from keras.layers import Activation
from keras.layers.convolutional import Convolution1D
from keras.layers.core import Dense, Dropout, Activation, Flatten


from keras.callbacks import ModelCheckpoint

from eeg_generator import EEGGenerator

def build_cnn(out_file):
    filters = 3
    filter_length = 3
    stride = 5
    n_elements = 240000
    n_ch = 16
    
    model = Sequential()
    model.add(Convolution1D(filters, filter_length, border_mode = 'same', input_shape = (n_elements, n_ch), activation = 'relu'))
    model.add(Convolution1D(filters, filter_length, border_mode = 'same', activation = 'relu', subsample_length = stride))
    print(model.output_shape)
    model.add(Convolution1D(filters, filter_length, border_mode = 'same', activation = 'relu'))
    model.add(Convolution1D(filters, filter_length, border_mode = 'same', activation = 'relu', subsample_length = stride))
    print(model.output_shape)
    model.add(Convolution1D(filters, filter_length, border_mode = 'same', activation = 'relu'))
    model.add(Convolution1D(filters, filter_length, border_mode = 'same', activation = 'relu', subsample_length = stride))
    print(model.output_shape)
    model.add(Convolution1D(filters, filter_length, border_mode = 'same', activation = 'relu'))
    model.add(Convolution1D(filters, filter_length, border_mode = 'same', activation = 'relu', subsample_length = stride))
    print(model.output_shape)
    model.add(Convolution1D(filters, filter_length, border_mode = 'same', activation = 'relu'))
    model.add(Convolution1D(filters, filter_length, border_mode = 'same', activation = 'relu', subsample_length = stride))
    print(model.output_shape)
    model.add(Flatten())
    print(model.output_shape)
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    print(model.output_shape)
    model.add(Dense(1))
    print(model.output_shape)
    model.add(Activation('sigmoid'))
    print(model.output_shape)
    
    model.compile(optimizer='adadelta',
              loss='binary_crossentropy',
              metrics=['accuracy'])
    
    model.save(out_file)
    
    del model
    
def main():
    model_dir = 'data/models/allcnn1'
    model_file = 'data/models/allcnn1/cnn.h5'
    best_weights_file = 'data/models/allcnn1/best_weights.h5'
    epoch_weights_file = 'data/models/allcnn1/epoch_{epoch:d}_weights.h5'
    if not isdir(model_dir):
        mkdir(model_dir)
    if not isfile(model_file):
        build_cnn(model_file)
    
    model = load_model(model_file)
    
    if isfile(best_weights_file):
        model.load_weights(best_weights_file)
    
    n_epochs = 10
    
    with EEGGenerator('data/train.h5') as gen:
        epoch_cp = ModelCheckpoint(epoch_weights_file, monitor = 'loss', save_weights_only = True)
        best_cp = ModelCheckpoint(best_weights_file, monitor = 'loss', save_weights_only = True, save_best_only = True)
        model.fit_generator(gen.gen_train(), gen.n_samples, n_epochs, class_weight = gen.class_weights, callbacks = [epoch_cp, best_cp])
    
if __name__ == '__main__':
    main()       
                
            