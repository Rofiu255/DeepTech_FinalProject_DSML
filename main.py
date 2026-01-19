from src.data_preprocessing import preprocess_data
from src.train_model import train_models
from src.evaluate_model import evaluate_model

if __name__ == "__main__":
    preprocess_data()
    train_models()
    evaluate_model()
