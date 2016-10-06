import h5py as h5
import numpy as np

data_k = 'data'
names_k = 'names'
classes_k = 'classes'
patients_k = 'patients'

class EEGReader:
    def __init__(self, data_file, val_split = 0.2, batch_size = 16):
        global data_k
        global classes_k
        self.hdf = h5.File(data_file, 'r')
        self.batch_size = batch_size
        
        self.n_samples = self.hdf[data_k].shape[0]
        self.n_batches = int((self.n_samples - 1) / self.batch_size) + 1
        n_train_batches = int(self.n_batches * (1 - val_split))
        n_val_batches = self.n_batches - n_train_batches
        self.n_train_samples = n_train_batches * self.batch_size
        self.n_val_samples = self.n_samples - self.n_train_samples
        
        np.random.seed(0)
        batch_order = np.random.permutation(self.n_batches)
        
        self.generators = {
            'train': TrainGenerator(self.hdf, 0, self.n_samples, n_train_batches, self.batch_size, batch_order),
            'val': ValidationGenerator(self.hdf, n_train_batches, self.n_samples, n_val_batches, self.batch_size, batch_order),
            'test': TestGenerator(self.hdf, 0, self.n_samples, self.batch_size),
            'patient': PatientGenerator(self.hdf, 0, self.n_samples, self.batch_size, val_split)
        }
        if classes_k in self.hdf:
            self.class_weights = np.histogram(self.hdf[classes_k][:], bins = 2, density = True)[0]
        else:
            self.class_weights = np.zeros(2)
        self.class_weights =  np.power(self.class_weights / self.class_weights.sum(), -1)
        self.class_weights = {0: self.class_weights[0], 1: self.class_weights[1]}
    
    def __enter__(self):
        return self
        
    def __exit__(self, a, b, c):
        self.hdf.close()
        return False
            
    def gen_train(self):
        return self.generators['train'].gen()
        
    def gen_val(self):
        return self.generators['val'].gen()
        
    def gen_test(self):
        return self.generators['test'].gen()
    
    def gen_patient(self):
        return self.generators['patient'].gen()
    
class TrainGenerator:
    def __init__(self, hdf, start_batch, n_samples, n_batches, batch_size, batch_order):
        self.hdf = hdf
        self.start_batch = start_batch
        self.n_samples = n_samples
        self.n_batches = n_batches
        self.batch_size = batch_size
        self.batch_order = batch_order
    
    def gen(self):
        global data_k
        global classes_k
        hdf = self.hdf
        start_batch = self.start_batch
        n_samples = self.n_samples
        n_batches = self.n_batches
        batch_size = self.batch_size
        
        seed = 0
        while True:
            np.random.seed(seed)
            train_order = np.random.permutation(n_batches)
            for batch in train_order:
                batch_start = self.batch_order[self.start_batch + batch] * batch_size
                batch_end = min(n_samples, batch_start + batch_size)
                n_batch_samples = batch_end - batch_start
                
                np.random.seed(batch)
                sample_order = np.random.permutation(n_batch_samples)
                
                x = hdf[data_k][batch_start:batch_end][sample_order]
                y = hdf[classes_k][batch_start:batch_end][sample_order]
                
                yield x, y
            seed += 1
class ValidationGenerator:
    def __init__(self, hdf, start_batch, n_samples, n_batches, batch_size, batch_order):
        self.hdf = hdf
        self.start_batch = start_batch
        self.n_samples = n_samples
        self.n_batches = n_batches
        self.batch_size = batch_size
        self.batch_order = batch_order
    
    def gen(self):
        global data_k
        global classes_k
        hdf = self.hdf
        start_batch = self.start_batch
        n_samples = self.n_samples
        n_batches = self.n_batches
        batch_size = self.batch_size
        
        while True:
            for batch in range(self.n_batches):
                batch_start = self.batch_order[self.start_batch + batch] * batch_size
                batch_end = min(n_samples, batch_start + batch_size)
                n_batch_samples = batch_end - batch_start
                
                x = hdf[data_k][batch_start:batch_end]
                y = hdf[classes_k][batch_start:batch_end]
                
                yield x, y
                
class TestGenerator:
    def __init__(self, hdf, start, end, batch_size):
        self.hdf = hdf
        self.start = start
        self.end = end
        self.batch_size = batch_size
    
    def gen(self):
        global data_k
        global names_k
        hdf = self.hdf
        start = self.start
        end = self.end
        batch_size = self.batch_size
        
        n_samples = end - start
        
        n_batches = int((n_samples - 1) / batch_size) + 1
        
        for batch in range(n_batches):
            batch_start = batch * batch_size + start
            batch_end = min(end, batch_start + batch_size)
            n_batch_samples = batch_end - batch_start
            
            x = hdf[data_k][batch_start:batch_end]
            names = hdf[names_k][batch_start:batch_end]
            
            yield names, x
            
class PatientGenerator:
    def __init__(self, hdf, start, end, batch_size, val_split):
        self.hdf = hdf
        self.start = start
        self.end = end
        self.batch_size = batch_size
        self.val_split = val_split
        
    def gen(self):
        hdf = self.hdf
        start = self.start
        end = self.end
        batch_size = self.batch_size
        val_split = self.val_split
        
        patient_ids = hdf[patients_k][start:end]
        unique, counts = np.unique(patient_ids, return_counts = True)
        n_patients = unique.shape[0]
        print(unique)
        print(counts)
        
        patient_start = 0
        for i in range(n_patients):
            patient_end = patient_start + counts[i]
            patient_samples = patient_end - patient_start
            patient_batches = int((patient_samples - 1) / batch_size) + 1
            patient_train_batches = int(patient_batches * (1 - val_split))
            patient_val_batches = patient_batches - patient_train_batches
            patient_train_samples = patient_train_batches * batch_size
            patient_val_samples = patient_samples - patient_train_samples
            
            np.random.seed(0)
            batch_order = np.random.permutation(patient_batches)
            
            generators = {
                'train': TrainGenerator(hdf, start, end, patient_train_batches, batch_size, batch_order),
                'val': ValidationGenerator(hdf, patient_train_batches, end, patient_val_batches, batch_size, batch_order),
                'test': TestGenerator(hdf, patient_start, patient_end, batch_size)
            }
            
            result = {
                'id': unique[i],
                'samples': patient_samples,
                'batches': patient_batches,
                'train_samples': patient_train_samples,
                'val_samples': patient_val_samples,
                'generators': generators
            }
            
            yield result
            
            patient_start += counts[i]