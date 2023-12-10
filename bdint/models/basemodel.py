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
