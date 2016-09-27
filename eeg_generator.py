import h5py as h5
import numpy as np

class EEGGenerator:
    data_key = 'data'
    names_key = 'names'
    classes_key = 'classes'
    
    def __init__(self, file_name, batch_size = 16):
        self.hdf = h5.File(file_name, 'r')
        self.batch_size = batch_size
        self.n_samples = self.hdf[self.names_key].shape[0]
        self.class_weights = np.histogram(self.hdf[self.classes_key][:], bins = 2, density = True)[0]
        self.class_weights =  np.power(self.class_weights / self.class_weights.sum(), -1)
        self.class_weights = {0: self.class_weights[0], 1: self.class_weights[1]}
        print(self.class_weights)
        
    def __enter__(self):
        return self
        
    def __exit__(self, type, value, traceback):
        print(type, value, traceback)
        return self
        
    def gen_train(self):
        while True:
            n_batches = int((self.n_samples - 1) / self.batch_size) + 1
            batch_order = np.random.permutation(n_batches)
            for i in range(n_batches):
                start = batch_order[i] * self.batch_size
                #print('%d/%d' % (i * self.batch_size, self.n_samples))
                end = min(self.n_samples, start + self.batch_size)
                train_x = self.hdf[self.data_key][start:end]
                train_y = self.hdf[self.classes_key][start:end]
                
                train_order = np.random.permutation(train_x.shape[0])
                
                yield train_x[train_order], train_y[train_order]
        
    def gen_test(self):
        n_batches = int((self.n_samples - 1) / self.batch_size) + 1
        for i in range(n_batches):
            start = i * self.batch_size
            print('%d/%d' % (start, self.n_samples))
            end = min(self.n_samples, start + self.batch_size)
            names = self.hdf[self.names_key][start:end]
            test_x = self.hdf[self.data_key][start:end]
            
            yield names, test_x