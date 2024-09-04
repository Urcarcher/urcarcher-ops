import pandas as pd
import numpy as np

from src.config.model_config import HISTORY_PATH
from sklearn.preprocessing import StandardScaler

COLUMNS = ["Date", "Price", "Open", "High", "Low", "Change %"]
COLUMNS_MATCHING = {"Price":0, "Open":1, "High":2, "Low":3}

class DataProcessor(object) :

    def __init__(
            self, train_by :str, 
            predict_target :str, 
            train_ratio :int,
            seq_len :int,
            pred_days :int,
            windows :int
    ) -> None :
        self._history = pd.read_csv(f"{HISTORY_PATH}/{train_by}.csv")

        self._seq_len = seq_len
        self._pred_days = pred_days
        self._windows = windows
        
        self._predict_target = predict_target
        self._y_true = self._history[predict_target].values
        self._dates = pd.to_datetime(self._history["Date"])
        self._historical_data = self._history[COLUMNS[1:]].astype(float)
        self._historical_data_scaled = None

        self._n_train = int(train_ratio * self._historical_data.shape[0])
        self._train_data_scaled = None
        self._train_dates = None

        self._test_data_scaled = None
        self._test_dates = None

        self._scaler = StandardScaler()

    def scaling(self) :
        self._scaler = self._scaler.fit(self._historical_data)
        self._historical_data_scaled = self._scaler.transform(self._historical_data)

        return self
    
    def set_train_test_data(self) :
        self._train_data_scaled = self._historical_data_scaled[:self._n_train]
        self._train_dates = self._dates[:self._n_train]

        self._test_data_scaled = self._historical_data_scaled[self._n_train:]
        self._test_dates = self._dates[self._n_train:]

        return self
    
    def get_data_set_by_sliding_window(self) -> list[np.ndarray] :
        
        trainX, trainY, testX, testY = [], [], [], []

        for i in range(self._seq_len, self._n_train-self._pred_days, self._windows):
            trainX.append(self._train_data_scaled[i-self._seq_len:i, COLUMNS_MATCHING[self._predict_target]])
            trainY.append(self._train_data_scaled[i:i+self._pred_days, COLUMNS_MATCHING[self._predict_target]])

        for i in range(self._seq_len, len(self._test_data_scaled)-self._pred_days+1, self._windows) :
            testX.append(self._test_data_scaled[i-self._seq_len:i, COLUMNS_MATCHING[self._predict_target]])
            testY.append(self._test_data_scaled[i:i+self._pred_days, COLUMNS_MATCHING[self._predict_target]])

        return [np.array(trainX), np.array(trainY), np.array(testX), np.array(testY)]

    def get_data_set_for_predict(self) -> np.ndarray :
        testX = []
        testX.append(self._train_data_scaled[self._n_train-self._seq_len:self._n_train, COLUMNS_MATCHING[self._predict_target]])
        
        return np.array(testX)
    
    def get_scaler(self) -> StandardScaler :
        return self._scaler