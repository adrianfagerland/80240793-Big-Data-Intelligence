from abc import ABC, abstractmethod

import numpy as np


class BaseModel(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def learn(self, x_train_df, y_train_df):
        pass

    @abstractmethod
    def predict(self, x_test_df) -> np.ndarray:
        pass
