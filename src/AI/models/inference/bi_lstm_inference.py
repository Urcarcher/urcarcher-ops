import numpy as np
import matplotlib.pyplot as plt

from src.AI.models.model_object import MultiLayeredBidirectionalLSTM
from src.AI.dataprocessing.history_processing import DataProcessor

class BiLSTMInference(object) :
    def __init__(
            self,
            model_config :dict[str, any]
    ) -> None :
        self._model_config = model_config
        self._model_path = model_config["model_path"]
        self._pred_days = model_config["last_dense_output"]

    def load_model(self) :
        self._model = MultiLayeredBidirectionalLSTM(
            self._model_config
        ).buildNN()

        self._model.load_weights(self._model_path)

    def get_1yr_predict(self) :
        data_processor = DataProcessor(
            "USD_KRW",
            "Open",
            1,
            1300,
            260,
            1300
        )

        data_processor.scaling().set_train_test_data()
        scaler = data_processor.get_scaler()

        testX = data_processor.get_data_set_for_predict()
        prediction = self._model.predict(testX)
        prediction = prediction.reshape(self._pred_days, -1)
        mean_values_pred = np.repeat(scaler.mean_[np.newaxis, :], prediction.shape[0], axis=0)
        mean_values_pred[:, 0] = np.squeeze(prediction)
        y_pred = scaler.inverse_transform(mean_values_pred)[:,0]

        print(y_pred)
        # plt.plot(y_pred[:],
        #         color='red',
        #         linestyle='--',
        #         label='Predicted Open Price')

        # plt.show()