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
        "model_path" : "src/AI/models/weights/BiLSTM_USD_KRW_Open.weights.h5"
    }
}

EXCHANGE_RATE_LIST = [
    "USD_KRW",
    "JPY_KRW",
    "CNY_KRW"
]

HISTORY_PATH = "src/AI/dataprocessing/origins"