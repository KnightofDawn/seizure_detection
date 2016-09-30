from keras.models import load_model
from pandas import DataFrame
from eeg_generator import EEGGenerator

def main():
    model_file = 'data/models/allcnn1/cnn.h5'
    weights_file = 'data/models/allcnn1/best_weights.h5'
    test_file = 'data/test.h5'
    prediction_file = 'data/models/allcnn1/prediction.csv'
    prediction_proba_file = 'data/models/allcnn1/prediction_proba.csv'
    
    model = load_model(model_file)
    weights = model.load_weights(weights_file)
    
    prediction_frame = DataFrame(columns = ['File', 'Class'])
    prediction_proba_frame = DataFrame(columns = ['File', 'Class'])
    
    with EEGGenerator(test_file) as gen:
        for names, test_x in gen.gen_test():
            prediction = model.predict_classes(test_x, batch_size = gen.batch_size, verbose = 0)
            prediction_proba = model.predict_proba(test_x, batch_size = gen.batch_size, verbose = 0)
            
            rows = [{'File': names[j], 'Class': prediction[j][0]} for j in range(len(names))]
            rows_proba = [{'File': names[j], 'Class': prediction_proba[j][0]} for j in range(len(names))]
            
            prediction_frame = prediction_frame.append(rows, ignore_index = True)
            prediction_proba_frame = prediction_proba_frame.append(rows_proba, ignore_index = True)
           
        prediction_frame.to_csv(prediction_file, index = False, cols = ['File', 'Class'], engine = 'python')
        prediction_proba_frame.to_csv(prediction_proba_file, index = False, cols = ['File', 'Class'], engine = 'python')
    
if __name__ == '__main__':
    main()