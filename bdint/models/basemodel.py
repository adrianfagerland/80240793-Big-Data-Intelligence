from abc import ABC, abstractmethod

import pandas as pd


class BaseModel(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def learn(self, x_train_df, y_train_df):
        pass

    @abstractmethod
    def predict(self, x_test_df):
        pass


def preprocess_for_numerical_model(df: pd.DataFrame) -> pd.DataFrame:
    preprocessed_df = df.copy().fillna(0).select_dtypes(include=["float64", "int64"])
    remove_columns = ["PoolQC", "Id"]
    remove_columns = [col for col in remove_columns if col in preprocessed_df.columns]
    return preprocessed_df.drop(columns=remove_columns)
