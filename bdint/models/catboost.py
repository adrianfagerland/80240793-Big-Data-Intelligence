import catboost as cb
from .basemodel import BaseModel


class CatBoost(BaseModel):
    def __init__(self, **kwargs):
        self.model = cb.CatBoostRegressor(**kwargs, random_seed=187)

    def learn(self, x_train_df, y_train_df):
        self.model.fit(x_train_df, y_train_df)

    def predict(self, x_test_df):
        return self.model.predict(x_test_df)
