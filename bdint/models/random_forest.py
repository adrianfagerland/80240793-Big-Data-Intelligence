from sklearn.ensemble import RandomForestRegressor
from sklearn.utils.validation import check_is_fitted

from .basemodel import BaseModel
from .utils import OHE


class RandomForest(BaseModel):
    categorical_columns = None

    def __init__(self, *args, **kwargs):
        self.model = RandomForestRegressor(**kwargs)
        self.ohe = OHE()

    def learn(self, x_train_df, y_train_df):
        x_train_df, self.categorical_columns = self.ohe.ohe(x_train_df)
        self.model.fit(x_train_df, y_train_df.values.ravel())

    def predict(self, x_test_df):
        check_is_fitted(self.model)
        x_test_df = self.ohe.ohe(x_test_df, self.categorical_columns)
        return self.model.predict(x_test_df)
