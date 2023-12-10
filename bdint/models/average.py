import pandas as pd

from bdint.models.basemodel import BaseModel


class AverageModel(BaseModel):
    mean = None

    def learn(self, x_train_df, y_train_df):
        self.mean = y_train_df["SalePrice"].mean()

    def predict(self, x_test_df):
        if self.mean is None:
            raise ValueError("Model has not been trained. Call 'learn' method first.")

        result_df = pd.DataFrame()
        result_df["Id"] = x_test_df["Id"]
        result_df["SalePrice"] = self.mean

        return result_df
