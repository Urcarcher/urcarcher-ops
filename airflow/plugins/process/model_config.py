MODEL_CONFIG = {
    "MultiLayeredBidirectionalLSTM" : {
        "input_shape" : (None, 1300, 1),
        "bidirectional_lstm_units" : 512,
        "gru_units" : 256,
        "first_dense_output" : 256,
        "last_dense_output" : 260,
        "drop_out_ratio" : 0.4,
        "batch_size" : 256,
        "learning_rate" : 0.0005,
        # "model_path" : "/mnt/c/close/shds/urcarcher/urcarcher-ops/airflow/plugins/process/models/weights/BiLSTM_USD_KRW_Open.weights.h5"
        "model_path" : "/mnt/c/close/shds/urcarcher/urcarcher-ops/airflow/plugins/process/models/weights"
    }
}

EXCHANGE_RATE_LIST = [
    "USD_KRW",
    "JPY_KRW",
    "CNY_KRW"
]

HISTORY_PATH = "/mnt/c/close/shds/urcarcher/urcarcher-ops/airflow/plugins/process/origins"

COLUMNS = ["Date", "Price", "Open", "High", "Low", "Change %"]