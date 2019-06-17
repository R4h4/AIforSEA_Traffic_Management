import pandas as pd
import numpy as np

from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, CuDNNLSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed

from tqdm import tqdm as tqdm


class Predictor:
    def __init__(self):
        self.model = self._load_model()
        self._load_weights()
        self._init_model()
        self.predictions = pd.DataFrame()

    def _load_model(self, filename='keras_model.json'):
        # Load json and create the model
        with open(filename, 'r') as json_file:
            model_json = json_file.read()
        return model_from_json(model_json)

    def _load_weights(self, filename='weights.h5'):
        self.model.load_weights(filename)

    def _init_model(self):
        self.model.compile(loss='mse', optimizer='adam')

    def predict(self, df):
        # Making sure the columns and rows are in the right order
        # assert df.columns == ['geohash6', 'demand', 'day', 'hour', 'weekday', 'total_demand', 'area_demand']
        predictions = []
        # Predict the demand for each geohash individually and write the predictions into a dict
        print("Predicting the demand for each geohash.s")
        for ghash in tqdm(df.geohash6.unique()):
            df_temp = df[df.geohash6 == ghash].copy()
            df_temp.drop('geohash6', axis=1, inplace=True)
            # Reshape the values to fit the model input shape
            input_x = df_temp.values
            input_x = input_x.reshape((1, input_x.shape[0], input_x.shape[1]))
            predictions.append([{'gehoash6': ghash,
                                 't+1': p[0][0].item(),
                                 't+2': p[1][0].item(),
                                 't+3': p[2][0].item(),
                                 't+4': p[3][0].item(),
                                 't+5': p[4][0].item()} for p in self.model.predict(input_x, verbose=0)])
        self.predictions = predictions

    def save_predictions(self, filename='predictions.csv'):
        pd.DataFrame(self.predictions).to_csv(filename, index=False)
