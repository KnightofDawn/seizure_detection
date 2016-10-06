import os
import errno

from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping

from sdetect.io import EEGReader
from sdetect.models import AllCNN

def make_dirs(file_name):
    if not os.path.exists(os.path.dirname(file_name)):
        try:
            os.makedirs(os.path.dirname(file_name))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
    

def main():
    train_file = 'data/train.h5'
    model_file = 'data/models/allcnn4/cnn.h5'
    
    make_dirs(model_file)
    
    allcnn = AllCNN()
    if(not os.path.exists(model_file)):
        model = allcnn.build_model()
        model.save(model_file)
    model = load_model(model_file)
    
    with EEGReader(train_file) as eeg_reader:
        for i, patient in enumerate(eeg_reader.gen_patient()):
            best_weights_file = 'data/models/allcnn4/weights%d/best_weights.h5' % patient['id']
            epoch_weights_file = 'data/models/allcnn4/weights%d/val_loss_{val_loss:.3f}_weights.h5' % patient['id']
            
            make_dirs(model_file)
            make_dirs(best_weights_file)
            make_dirs(epoch_weights_file)
            
            if(os.path.exists(best_weights_file)):
                model.load_weights(best_weights_file)
            
            epoch_cp = ModelCheckpoint(epoch_weights_file, monitor = 'val_loss', save_weights_only = True)
            best_cp = ModelCheckpoint(best_weights_file, monitor = 'val_loss', save_weights_only = True, save_best_only = True)
            early_stopping = EarlyStopping(monitor='val_loss', patience = 10)
            
            n_epochs = 100
            samples_per_epoch = patient['samples']
            model.fit_generator(patient['generators']['train'].gen(), samples_per_epoch, n_epochs, validation_data = patient['generators']['val'].gen(), nb_val_samples = patient['val_samples'], class_weight = eeg_reader.class_weights, callbacks = [epoch_cp, best_cp, early_stopping])  
            
if __name__ == '__main__':
    main()