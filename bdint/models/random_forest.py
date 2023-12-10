import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils.validation import check_is_fitted

from bdint.models.basemodel import BaseModel, preprocess_for_numerical_model


class RandomForest(BaseModel):
    def __init__(self, *args, **kwargs):
        self.model = RandomForestRegressor(**kwargs)

    def learn(self, x_train_df, y_train_df):
        x_train_df = preprocess_for_numerical_model(x_train_df)
        self.model.fit(x_train_df, y_train_df.values.ravel())

    def predict(self, x_test_df):
        check_is_fitted(self.model)
        x_test_df = preprocess_for_numerical_model(x_test_df)
        return self.model.predict(x_test_df)
