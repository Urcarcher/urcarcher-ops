import numpy as np
import matplotlib.pyplot as plt

from process.models.model_object import MultiLayeredBidirectionalLSTM
from process.database import UrcarcherDBManager
from process.data import DataProcessor
from process.data import get_next_1yr
from process.model_config import COLUMNS
from process.model_config import EXCHANGE_RATE_LIST

class BiLSTMInference(object) :
    def __init__(
            self,
            model_config :dict[str, any]
    ) -> None :
        self._model_config = model_config
        self._model_path = model_config["model_path"]
        self._pred_days = model_config["last_dense_output"]

        self._db_manager = UrcarcherDBManager()

        self._type_dict = {}

        for type in EXCHANGE_RATE_LIST :
            self._type_dict[type] = None

    def _load_model(self, exchange_type :str, column :str) -> MultiLayeredBidirectionalLSTM :
        model = MultiLayeredBidirectionalLSTM(
            self._model_config
        ).buildNN()

        model.load_weights(f"{self._model_path}/BiLSTM_{exchange_type}_{column}.weights.h5")

        return model

    def get_1yr_predict(self) -> str :
        new_data = get_next_1yr()

        for type in EXCHANGE_RATE_LIST :
            for col in COLUMNS[1:5] :
                data_processor = DataProcessor(
                    type,
                    col,
                    1,
                    1300,
                    260,
                    1300
                )

                data_processor.scaling().set_train_test_data()
                scaler = data_processor.get_scaler()

                testX = data_processor.get_data_set_for_predict()
                prediction = self._load_model(type, col).predict(testX)
                prediction = prediction.reshape(self._pred_days, -1)
                mean_values_pred = np.repeat(scaler.mean_[np.newaxis, :], prediction.shape[0], axis=0)
                mean_values_pred[:, 0] = np.squeeze(prediction)
                y_pred = scaler.inverse_transform(mean_values_pred)[:,0]

                new_data[col] = y_pred

            self._type_dict[type] = new_data

        # plt.plot(y_pred[:],
        #         color='red',
        #         linestyle='--',
        #         label='Predicted Open Price')

        # plt.show()

        return "predict success."
    
    def update_db(self) :
        for type in EXCHANGE_RATE_LIST :
            self._db_manager.update_db(self._type_dict[type], type)

        return "db update success."