from abc import ABC, abstractmethod


class BaseModel(ABC):
    @abstractmethod
    def learn(self, x_train_df, y_train_df):
        pass

    @abstractmethod
    def predict(self, x_test_df):
        pass
