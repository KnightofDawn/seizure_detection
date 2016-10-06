import os

from keras.models import load_model
from pandas import DataFrame
import numpy as np

from sdetect.io import EEGReader
from sdetect.models import AllCNN

def main():
    test_file = 'data/test.h5'
    prediction_file = 'data/models/allcnn4/prediction.csv'
    model_file = 'data/models/allcnn4/cnn.h5'
    model = load_model(model_file)
    
    allcnn = AllCNN()
    
    with EEGReader(test_file) as eeg_reader:
        idx = 0
        name_arr = np.array([''] * eeg_reader.n_samples, dtype = object)
        prediction_arr = np.zeros(eeg_reader.n_samples)
        for i, patient in enumerate(eeg_reader.gen_patient()):
            weights_file = 'data/models/allcnn4/weights%d/best_weights.h5' % int(patient['id'])
            model.load_weights(weights_file)
            
            for names, test_x in patient['generators']['test'].gen():
                print('%d/%d' % (idx, eeg_reader.n_samples))
                batch_samples = len(names)
                prediction = model.predict_proba(test_x, batch_size = batch_samples, verbose = 0)
                prediction_arr[idx:idx + batch_samples] = prediction.flatten()
                name_arr[idx:idx + batch_samples] = names
                
                idx = idx + batch_samples
                
                for name in names:
                    patient_id = int(os.path.basename(name).split('.')[-2].split('_')[0])
                    if(not patient_id == int(patient['id'])):
                        print(patient['id'], name)
        prediction_frame = DataFrame(data = {'File': name_arr, 'Class': prediction_arr})
        prediction_frame.to_csv(prediction_file, index = False, cols = ['File', 'Class'], engine = 'python')
            
if __name__ == '__main__':
    main()