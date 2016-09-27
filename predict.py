from keras.models import load_model
from pandas import DataFrame
from eeg_generator import EEGGenerator

def main():
    model_file = 'data/models/allcnn1/cnn.h5'
    weights_file = 'data/models/allcnn1/epoch_1_weights.h5'
    test_file = 'data/test.h5'
    prediction_file = 'data/models/allcnn1/prediction.csv'
    
    model = load_model(model_file)
    weights = model.load_weights(weights_file)
    
    prediction_frame = DataFrame(columns = ['File', 'Prediction'])
    
    with EEGGenerator(test_file) as gen:
        for names, test_x in gen.gen_test():
            prediction = model.predict_proba(test_x, batch_size = gen.batch_size)
            rows = [{'File': names[j], 'Class': prediction[j]} for j in range(len(names))]
            prediction_frame = prediction_frame.append(rows, ignore_index = True)
    
        prediction_frame.to_csv(prediction_file, index = False)
    
if __name__ == '__main__':
    main()