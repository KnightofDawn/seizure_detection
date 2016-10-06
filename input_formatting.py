import os
import glob

import h5py as h5
import numpy as np

from scipy.io import loadmat

def load_eeg(mat_file):
    data = loadmat(mat_file)
    return data['dataStruct'][0][0][0]

def extract_train_data():
    print('extracting train data...')

    in_dir = 'data/train'
    out_file = 'data/train.h5'
    
    data_key = 'data'
    classes_key = 'classes'
    patients_key = 'patients'
    
    mat_files = glob.glob('%s/*/*.mat' % in_dir)
    classes = [int(os.path.basename(mat_file).split('.')[-2].split('_')[-1]) for mat_file in mat_files]
    patients = [int(os.path.basename(mat_file).split('.')[-2].split('_')[0]) for mat_file in mat_files]
    
    n_samples = len(mat_files)
    n_elements = 240000
    n_ch = 16
    
    with h5.File(out_file, 'w') as hdf:
        hdf.create_dataset(data_key, (n_samples, n_elements, n_ch))
        hdf.create_dataset(classes_key, data = classes)
        hdf.create_dataset(patients_key, data = patients)
        
        for i in range(n_samples):
            if(i % 32 == 0):
                print('%d/%d' % (i, n_samples))
            data = load_eeg(mat_files[i])
            hdf[data_key][i] = load_eeg(mat_files[i])
            
def extract_test_data():
    print('extracting test data...')

    in_dir = 'data/test'
    out_file = 'data/test.h5'
    
    data_key = 'data'
    names_key = 'names'
    patients_key = 'patients'
    
    mat_files = glob.glob('%s/*/*.mat' % in_dir)
    names = [os.path.basename(mat_file)for mat_file in mat_files]
    patients = [name.split('.')[-2].split('_')[0] for name in names]
    
    n_samples = len(mat_files)
    n_elements = 240000
    n_ch = 16
    
    with h5.File(out_file, 'w') as hdf:
        hdf.create_dataset(data_key, (n_samples, n_elements, n_ch))
        hdf.create_dataset(names_key, data = names)
        hdf.create_dataset(patients_key, data = patients)
        
        for i in range(n_samples):
            if(i % 32 == 0):
                print('%d/%d' % (i, n_samples))
            print(hdf[data_key][i].shape)
            hdf[data_key][i] = load_eeg(mat_files[i])
            
def main():
    extract_train_data()
    extract_test_data()
    
if __name__ == '__main__':
    main()