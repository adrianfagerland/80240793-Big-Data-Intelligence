import numpy as np

from .basemodel import BaseModel


class AveragePredicter(BaseModel):
    def __init__(self):
        self.mean = None

    def learn(self, x_train_df, y_train_df):
        self.mean = y_train_df["SalePrice"].mean()

    def predict(self, x_test_df):
        if self.mean is None:
            raise ValueError("Model has not been trained. Call 'learn' method first.")

        prediction = np.ndarray(shape=(len(x_test_df)))
        prediction.fill(self.mean)

        return prediction
