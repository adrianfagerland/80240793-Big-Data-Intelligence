from sklearn.ensemble import RandomForestRegressor
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import mean_squared_error

from bdint.models.basemodel import BaseModel

from bdint.models.utils import ohe



class RandomForest(BaseModel):
    def __init__(self, *args, **kwargs):
        self.model = RandomForestRegressor(n_estimators = 100, random_state = 187)

    def learn(self, x_train_df, y_train_df):
        #categoricals have to be OHE
        self.model.fit(x_train_df, y_train_df.values.ravel())

    def predict(self, x_test_df):
        check_is_fitted(self.model)
        #x_test_df = preprocess_for_numerical_model(x_test_df)
        return self.model.predict(x_test_df)
