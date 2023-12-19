from catboost import CatBoostRegressor, Pool, cv
from sklearn.model_selection import train_test_split

from .basemodel import BaseModel


class CatBoost(BaseModel):
    def __init__(self, **kwargs):
        self.model = CatBoostRegressor(verbose=250, **kwargs)

    def _preprocess(self, df):
        df = df.fillna("NA")
        cat_features = df.select_dtypes(include=["object"]).columns
        df[cat_features] = df[cat_features].astype(str)
        cat_features_indices = [df.columns.get_loc(c) for c in cat_features]
        return df, cat_features_indices

    def cross_validation(self, x_train_df, y_train_df, folds=5):
        x_train_df, cat_features_indices = self._preprocess(x_train_df)

        pool = Pool(
            x_train_df,
            label=y_train_df,
            cat_features=cat_features_indices,
        )
        params = self.model.get_params()  # define different parameters to test here
        scores = cv(pool, params, fold_count=folds, return_models=True, plot=True)
        return scores

    def learn(self, x_train_df, y_train_df, validate=True):
        x_train_df, cat_features_indices = self._preprocess(x_train_df)
        x_train, x_val, y_train, y_val = train_test_split(
            x_train_df, y_train_df, test_size=0.05
        )
        self.model.fit(
            x_train,
            y_train,
            cat_features=cat_features_indices,
            eval_set=((x_val, y_val) if validate else None),
        )
        # return the rmse from the eval set
        if validate:
            return self.model.get_best_score()["validation"]["RMSE"]

    def save_model(self, path):
        self.model.save_model(path)

    def load_model(self, path):
        self.model.load_model(path)

    def predict(self, x_test_df):
        x_test_df, _ = self._preprocess(x_test_df)
        return self.model.predict(x_test_df)
