from src.AI.models.inference import BiLSTMInference

from src.config.model_config import MODEL_CONFIG

if __name__ == "__main__" :

    inference = BiLSTMInference(
        MODEL_CONFIG["MultiLayeredBidirectionalLSTM"]
    )

    inference.load_model()

    inference.get_1yr_predict()
    