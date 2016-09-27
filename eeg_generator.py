import h5py as h5

class EEGGenerator:
    data_key = 'data'
    names_key = 'names'
    classes_key = 'classes'
    
    def __init__(self, file_name, batch_size = 32):
        self.hdf = h5.File(file_name, 'r')
        self.batch_size = batch_size
        self.n_samples = self.hdf[self.names_key].shape[0]
        
    def __enter__(self):
        return self
        
    def __exit__(self, type, value, traceback):
        print(type, value, traceback)
        return self
        
    def gen_test(self):
        n_batches = int((self.n_samples - 1) / self.batch_size) + 1
        for i in range(n_batches):
            start = i * self.batch_size
            end = min(self.n_samples, start + self.batch_size)
            names = self.hdf[self.names_key][start:end]
            test_x = self.hdf[self.data_key][start:end]
            
            yield names, test_x