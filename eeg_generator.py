import h5py as h5
import numpy as np

class EEGGenerator:
    data_key = 'data'
    names_key = 'names'
    classes_key = 'classes'
    
    def __init__(self, file_name, batch_size = 8, split = 0.1):
        self.hdf = h5.File(file_name, 'r')
        self.batch_size = batch_size
        self.n_samples = self.hdf[self.names_key].shape[0]
        train_end = int((1 - split) * self.n_samples)
        self.n_batches = int((self.n_samples - 1) / self.batch_size) + 1
        self.n_train_batches = int((train_end - 1) / self.batch_size) + 1
        self.n_val_samples = (self.n_samples - (self.n_train_batches * self.batch_size))
        
        np.random.seed(0)
        self.batch_order = np.random.permutation(self.n_batches)
        
        self.class_weights = np.histogram(self.hdf[self.classes_key][:], bins = 2, density = True)[0]
        self.class_weights =  np.power(self.class_weights / self.class_weights.sum(), -1)
        self.class_weights = {0: self.class_weights[0], 1: self.class_weights[1]}

    def __enter__(self):
        return self
        
    def __exit__(self, type, value, traceback):
        return False
    
    def gen_train(self):
        while True:
            train_batch_order = np.random.permutation(self.n_train_batches)
            for i in range(self.n_train_batches):
                start = self.batch_order[train_batch_order[i]] * self.batch_size
                end = min(self.n_samples, start + self.batch_size)
                train_x = self.hdf[self.data_key][start:end]
                train_y = self.hdf[self.classes_key][start:end]
                
                train_order = np.random.permutation(train_x.shape[0])
                
                yield train_x[train_order], train_y[train_order]
    
    def gen_val(self):
        while True:
            for i in range(self.n_train_batches, self.n_batches):
                print('%d/%d' % ((i - self.n_train_batches) * self.batch_size, self.n_samples - (self.n_train_batches * self.batch_size)))
                start = self.batch_order[i] * self.batch_size
                end = min(self.n_samples, start + self.batch_size)
                val_x = self.hdf[self.data_key][start:end]
                val_y = self.hdf[self.classes_key][start:end]
                
                yield val_x, val_y
        
    def gen_test(self):
        for i in range(self.n_batches):
            start = i * self.batch_size
            print('%d/%d' % (start, self.n_samples))
            end = min(self.n_samples, start + self.batch_size)
            names = self.hdf[self.names_key][start:end]
            test_x = self.hdf[self.data_key][start:end]
            
            yield names, test_x