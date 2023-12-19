from sklearn.kernel_ridge import KernelRidge

from .basemodel import BaseModel


class LinearRegross(BaseModel):
    def __init__(self, **kwargs):
        self.model = KernelRidge(**kwargs)

    def learn(self, x_train_df, y_train_df):
        self.model.fit(x_train_df, y_train_df.values.ravel())

    def predict(self, x_test_df):
        return self.model.predict(x_test_df)
